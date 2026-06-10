from __future__ import annotations

import atexit
import math
import time
from dataclasses import fields, replace
import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer
import numpy as np

from babyvllm.config import Config
from babyvllm.engine.sequence import Sequence
from babyvllm.engine.scheduler import Scheduler
from babyvllm.engine.model_runner import ModelRunner
from babyvllm.sampling_params import SamplingParams

def worker_process(config, rank, event):
    """ Entrance of worker process. Create a model runner and enters loop. """
    # In multiprocessing, print of worker process may store in buffer,
    # which may cause some log messages to be lost.
    # To avoid this, set line buffering for stdout and stderr.
    import sys
    import os
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)
    
    model_runner = ModelRunner(config, rank, event)
    model_runner.loop()

class LLMEngine:
    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k:v for k, v in kwargs.items() if k in config_fields}
        self.config = Config(model, **config_kwargs)
        config = self.config
        
        # Create and start multiple worker processes.
        # Get Pytorch multiprocessing context and use `spawn` mode instead of 'fork' mode to create worker processes.
        ctx = mp.get_context("spawn")
        # `processes` stores all worker processes and is used to manage them, such as wait for them to finish.
        self.processes = []
        # `events` stores all events that are used to communicate between the main process and worker processes.
        self.events = []
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=worker_process, args=(config, i, event))
            self.processes.append(process)
            self.events.append(event)
            process.start()
        # Model runner in the main process will coordinate model execution in worker processes.
        self.model_runner = ModelRunner(config, rank=0, event=self.events)
        
        # Tokenizer will convert input text into token ids before model execution,
        # and convert token ids back to text after model execution.
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        
        # Adapt the block size of Sequence to the kv cache block size.
        Sequence.block_size = config.kvcache_block_size
        
        # Construct scheduler after model runner, because scheduler needs to know the number of kv cache blocks.
        self.scheduler = Scheduler(config)
    
        # When the program exits, clean up resources automatically.
        atexit.register(self.exit)
        
    def exit(self):
        """ Clean up resources when the program exits. """

        if hasattr(self, 'model_runner') and self.model_runner is not None:
            self.model_runner.call('exit')
            del self.model_runner
        # Wait for all worker processes to finish.
        for process in getattr(self, 'processes', []):
            process.join()
        self.processes = []

    def _validate_sampling_params(self, sampling_params: SamplingParams):
        if not isinstance(sampling_params, SamplingParams):
            raise ValueError("sampling_params must be a SamplingParams instance or a list of SamplingParams.")
        if not isinstance(sampling_params.max_tokens, int) or isinstance(sampling_params.max_tokens, bool) or sampling_params.max_tokens <= 0:
            raise ValueError("sampling_params.max_tokens must be a positive integer.")
        if sampling_params.max_model_length is not None:
            if (
                not isinstance(sampling_params.max_model_length, int)
                or isinstance(sampling_params.max_model_length, bool)
                or sampling_params.max_model_length <= 0
            ):
                raise ValueError("sampling_params.max_model_length must be a positive integer or None.")
            if sampling_params.max_model_length > self.config.max_model_length:
                raise ValueError(
                    f"sampling_params.max_model_length ({sampling_params.max_model_length}) "
                    f"exceeds engine max_model_length ({self.config.max_model_length})."
                )

    def _validate_prompt_token_ids(self, prompt_token_ids: list[int]):
        if not isinstance(prompt_token_ids, list):
            raise ValueError("prompt_token_ids must be a list[int].")
        if not prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty.")
        if any(not isinstance(token_id, int) or isinstance(token_id, bool) for token_id in prompt_token_ids):
            raise ValueError("prompt_token_ids must contain only integers.")

    def _coerce_prompt_token_ids(
        self,
        prompt: str = None,
        prompt_token_ids: list[int] = None,
    ) -> list[int]:
        if (prompt is None) == (prompt_token_ids is None):
            raise ValueError("Exactly one of prompt or prompt_token_ids must be provided.")
        if prompt is not None:
            if not isinstance(prompt, str):
                raise ValueError("prompt must be a string.")
            prompt_token_ids = self.tokenizer.encode(prompt)
        self._validate_prompt_token_ids(prompt_token_ids)
        return list(prompt_token_ids)

    def _effective_max_model_length(self, sampling_params: SamplingParams) -> int:
        effective_max_model_length = self.config.max_model_length
        if self.config.num_kvcache_blocks > 0:
            kv_cache_token_capacity = self.config.num_kvcache_blocks*self.config.kvcache_block_size
            effective_max_model_length = min(effective_max_model_length, kv_cache_token_capacity)
        if sampling_params.max_model_length is not None:
            effective_max_model_length = min(effective_max_model_length, sampling_params.max_model_length)
        return effective_max_model_length

    def _prepare_request(
        self,
        sampling_params: SamplingParams,
        prompt: str = None,
        prompt_token_ids: list[int] = None,
    ) -> tuple[list[int], SamplingParams]:
        self._validate_sampling_params(sampling_params)
        prompt_token_ids = self._coerce_prompt_token_ids(prompt=prompt, prompt_token_ids=prompt_token_ids)
        effective_max_model_length = self._effective_max_model_length(sampling_params)

        prompt_len = len(prompt_token_ids)
        if prompt_len > self.config.max_num_batched_tokens:
            raise ValueError(
                f"Prompt length ({prompt_len}) exceeds max_num_batched_tokens "
                f"({self.config.max_num_batched_tokens})."
            )
        if self.config.num_kvcache_blocks > 0:
            prompt_blocks = math.ceil(prompt_len/self.config.kvcache_block_size)
            if prompt_blocks > self.config.num_kvcache_blocks:
                raise ValueError(
                    f"Prompt requires {prompt_blocks} KV cache blocks, but only "
                    f"{self.config.num_kvcache_blocks} blocks are available."
                )
        if prompt_len >= effective_max_model_length:
            raise ValueError(
                f"Prompt length ({prompt_len}) must be smaller than effective max_model_length "
                f"({effective_max_model_length}) to leave room for completion tokens."
            )

        return prompt_token_ids, replace(sampling_params, max_model_length=effective_max_model_length)

    def _enqueue_request(self, prompt_token_ids: list[int], sampling_params: SamplingParams):
        seq = Sequence(token_ids=prompt_token_ids, sampling_params=sampling_params)
        self.scheduler.add_sequence(seq)
        return seq

    def add_request(
        self,
        sampling_params: SamplingParams,
        prompt: str = None,
        prompt_token_ids: list[int] = None,
    ) -> int:
        """
        Add input prompt to the scheduler's waiting queue.
        Support both `string` format and `list[int]` format.
        Return the number of tokens in the prompt.
        """

        prompt_token_ids, sampling_params = self._prepare_request(
            sampling_params=sampling_params,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
        )
        self._enqueue_request(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
        return len(prompt_token_ids)

    def step(self) -> tuple[list[int], int, bool]:
        """ Run the model for scheduled sequences. """
        
        # (1) Schedule sequences.
        scheduled_sequences, is_prefill = self.scheduler.schedule()
        # There is no sequence to schedule.
        if not scheduled_sequences:
            return [], 0, is_prefill
        
        # (2) Run the model.
        outputs = self.model_runner.call('run', scheduled_sequences, is_prefill)
        
        # (3) Postprocess the model outputs.
        self.scheduler.postprocess(scheduled_sequences, outputs)
        
        # (4) Collect finished sequences.
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in scheduled_sequences if seq.is_finished]
        num_processed_tokens = sum(len(seq) for seq in scheduled_sequences) if is_prefill else len(scheduled_sequences)
        
        return outputs, num_processed_tokens, is_prefill

    def generate(
        self,
        prompts: list[str],
        sampling_params: SamplingParams,
    ) -> list[str]:
        """ Generate text for all input prompts. """

        if not isinstance(prompts, list):
            raise ValueError("prompts must be a list of strings or a list of token ID lists.")
        if isinstance(sampling_params, list):
            if len(sampling_params) != len(prompts):
                raise ValueError("When sampling_params is a list, its length must match prompts.")
            sampling_params_list = sampling_params
        else:
            sampling_params_list = [sampling_params]*len(prompts)

        prepared_requests = []
        for prompt_data, sp in zip(prompts, sampling_params_list):
            if isinstance(prompt_data, str):
                prepared_requests.append(self._prepare_request(sampling_params=sp, prompt=prompt_data))
            elif isinstance(prompt_data, list):
                prepared_requests.append(self._prepare_request(sampling_params=sp, prompt_token_ids=prompt_data))
            else:
                raise ValueError("Prompts must be a list of strings or a list of token ID lists.")

        # Add all input prompts to the scheduler's waiting queue.
        prompt_tokens_cnt = 0
        for prompt_token_ids, sp in prepared_requests:
            self._enqueue_request(prompt_token_ids=prompt_token_ids, sampling_params=sp)
            prompt_tokens_cnt += len(prompt_token_ids)
                
        # {sequence id : generated tokens}
        generated_tokens = {}
        
        # Metrics collection
        total_time = 0
        memory_usages = []
        gpu_utilizations = []
        inference_start_time = time.time()
        first_token_times = {}
        sequence_start_times = {}
        sequence_token_counts = {}
        
        while not self.scheduler.is_finished():
            start_time = time.time()
            # Call scheduler and model runner to run the model.
            # outputs: {sequence id : generated tokens}
            outputs, num_processed_tokens, is_prefill = self.step()
            if num_processed_tokens == 0 and not self.scheduler.is_finished():
                raise RuntimeError("Scheduler made no progress while unfinished requests remain.")
            end_time = time.time()
            
            # Collect metrics
            step_time = end_time-start_time
            total_time += step_time
            
            # Get memory usage
            memory_stats = torch.cuda.memory_stats()
            memory_usage = memory_stats['allocated_bytes.all.current']/(1024**2)  # Convert to MB
            memory_usages.append(memory_usage)
            
            # Get GPU utilization
            gpu_util = torch.cuda.utilization()
            gpu_utilizations.append(gpu_util)

            # Update generated tokens and track TTFT
            for seq_id, tokens in outputs:
                if seq_id not in sequence_start_times:
                    sequence_start_times[seq_id] = inference_start_time
                if seq_id not in first_token_times and len(tokens) > 0:
                    first_token_times[seq_id] = time.time()
                sequence_token_counts[seq_id] = len(tokens)
                generated_tokens[seq_id] = tokens
                
        # Sort generated tokens by sequence id. So, the output text are in the same order as user input.
        generated_tokens = [generated_tokens[seq_id] for seq_id in sorted(generated_tokens.keys())]
        
        # Calculate metrics
        inference_end_time = time.time()
        total_inference_time = inference_end_time-inference_start_time
        generated_tokens_cnt = sum(sequence_token_counts.values())
        total_tokens_cnt = prompt_tokens_cnt+generated_tokens_cnt
        average_throughput = total_tokens_cnt/total_inference_time if total_inference_time > 0 else 0
        average_memory = sum(memory_usages)/len(memory_usages) if memory_usages else 0
        average_gpu_util = sum(gpu_utilizations)/len(gpu_utilizations) if gpu_utilizations else 0

        ttft_values = []
        tpot_values = []
        for seq_id in sequence_start_times:
            if seq_id in first_token_times:
                ttft = first_token_times[seq_id] - sequence_start_times[seq_id]
                ttft_values.append(ttft)
                
                if seq_id in sequence_token_counts and sequence_token_counts[seq_id] > 0:
                    # TPOT = (total generation time - TTFT) / number of tokens
                    total_gen_time = inference_end_time - first_token_times[seq_id]
                    tpot = total_gen_time / sequence_token_counts[seq_id] if sequence_token_counts[seq_id] > 0 else 0
                    tpot_values.append(tpot)

        metrics = {
            "total_tokens": total_tokens_cnt,
            "total_time": total_inference_time,
            "throughput": average_throughput,
            "avg_memory_mb": average_memory,
            "avg_gpu_util": average_gpu_util,
            "ttft": {
                "avg": np.mean(ttft_values) if ttft_values else 0,
                "p50": np.percentile(ttft_values, 50) if ttft_values else 0,
                "p90": np.percentile(ttft_values, 90) if ttft_values else 0,
                "p99": np.percentile(ttft_values, 99) if ttft_values else 0,
            },
            "tpot": {
                "avg": np.mean(tpot_values) if tpot_values else 0,
                "p50": np.percentile(tpot_values, 50) if tpot_values else 0,
                "p90": np.percentile(tpot_values, 90) if tpot_values else 0,
                "p99": np.percentile(tpot_values, 99) if tpot_values else 0,
            }
        }
        
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in generated_tokens]
        
        return outputs, metrics
