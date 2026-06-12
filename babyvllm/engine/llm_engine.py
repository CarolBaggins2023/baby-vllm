from __future__ import annotations

import atexit
import math
import traceback
import time
from dataclasses import fields, replace
import torch
import torch.multiprocessing as mp
from multiprocessing.connection import Connection
from transformers import AutoTokenizer
import numpy as np

from babyvllm.config import Config
from babyvllm.engine.sequence import Sequence
from babyvllm.engine.scheduler import Scheduler
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

    from babyvllm.engine.model_runner import ModelRunner
    model_runner = ModelRunner(config, rank, event)
    model_runner.loop()


def data_parallel_worker_process(model, config_kwargs, connection: Connection):
    """Run one full offline engine replica for a DP rank."""
    import sys
    import os

    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

    rank = config_kwargs.get("data_parallel_rank", 0)
    llm = None
    try:
        llm = LLMEngine(model, **config_kwargs)
        connection.send(("ready", rank))
        while True:
            message = connection.recv()
            if not isinstance(message, tuple) or not message:
                connection.send(("error", None, rank, "Malformed DP worker message."))
                continue
            command = message[0]
            if command == "exit":
                break
            if command != "generate" or len(message) != 5:
                connection.send(("error", None, rank, f"Unknown DP worker command: {command!r}"))
                continue

            _, request_id, indices, prompts, sampling_params_list = message
            try:
                outputs, metrics = llm.generate(prompts, sampling_params_list)
                connection.send(("result", request_id, rank, indices, outputs, metrics))
            except BaseException:
                connection.send(("error", request_id, rank, traceback.format_exc()))
    except BaseException:
        try:
            connection.send(("error", None, rank, traceback.format_exc()))
        except Exception:
            pass
    finally:
        if llm is not None:
            llm.exit()
        connection.close()

class LLMEngine:
    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k:v for k, v in kwargs.items() if k in config_fields}
        self.config = Config(model, **config_kwargs)
        config = self.config
        self._model = model
        self._exited = False
        self._config_kwargs = {
            field.name: getattr(config, field.name)
            for field in fields(Config)
            if field.name not in {"model", "hf_config", "num_kvcache_blocks"}
        }
        self._is_data_parallel_coordinator = config.data_parallel_size > 1

        if self._is_data_parallel_coordinator:
            self._init_data_parallel_workers()
            atexit.register(self.exit)
            return

        self._init_single_replica()
        atexit.register(self.exit)

    def _init_single_replica(self):
        from babyvllm.engine.model_runner import ModelRunner

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

    def _worker_config_kwargs(self, rank: int) -> dict:
        config_kwargs = dict(self._config_kwargs)
        config_kwargs.update(
            data_parallel_size=1,
            data_parallel_world_size=self.config.effective_data_parallel_size,
            data_parallel_rank=rank,
            distributed_init_method=None,
            shared_memory_name=None,
            num_kvcache_blocks=-1,
        )
        return config_kwargs

    def _init_data_parallel_workers(self):
        ctx = mp.get_context("spawn")
        self.dp_processes = []
        self.dp_connections = []
        self._dp_request_id = 0

        for rank in range(self.config.data_parallel_size):
            parent_conn, child_conn = ctx.Pipe()
            process = ctx.Process(
                target=data_parallel_worker_process,
                args=(self._model, self._worker_config_kwargs(rank), child_conn),
                name=f"babyvllm_dp_rank_{rank}",
            )
            process.start()
            child_conn.close()
            self.dp_processes.append(process)
            self.dp_connections.append(parent_conn)

        try:
            for rank in range(self.config.data_parallel_size):
                self._wait_for_worker_ready(rank)
        except BaseException:
            self.exit()
            raise

    def _wait_for_worker_ready(self, rank: int):
        message = self._recv_data_parallel_message(rank, request_id=None)
        if not isinstance(message, tuple) or message != ("ready", rank):
            raise RuntimeError(f"Malformed ready message from DP rank {rank}: {message!r}")
        
    def exit(self):
        """ Clean up resources when the program exits. """
        if self._exited:
            return
        self._exited = True

        if getattr(self, "_is_data_parallel_coordinator", False):
            self._shutdown_data_parallel_workers()
            return

        if hasattr(self, 'model_runner') and self.model_runner is not None:
            self.model_runner.call('exit')
            del self.model_runner
        # Wait for all worker processes to finish.
        for process in getattr(self, 'processes', []):
            process.join()
        self.processes = []

    def _shutdown_data_parallel_workers(self):
        for connection, process in zip(
            getattr(self, 'dp_connections', []),
            getattr(self, 'dp_processes', []),
        ):
            if process.is_alive():
                try:
                    connection.send(("exit",))
                except (BrokenPipeError, EOFError, OSError):
                    pass
        for connection, process in zip(
            getattr(self, 'dp_connections', []),
            getattr(self, 'dp_processes', []),
        ):
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join()
            connection.close()
        self.dp_processes = []
        self.dp_connections = []

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

    def _normalize_generation_inputs(
        self,
        prompts: list[str],
        sampling_params: SamplingParams,
    ) -> list[SamplingParams]:
        if not isinstance(prompts, list):
            raise ValueError("prompts must be a list of strings or a list of token ID lists.")
        if isinstance(sampling_params, list):
            if len(sampling_params) != len(prompts):
                raise ValueError("When sampling_params is a list, its length must match prompts.")
            sampling_params_list = sampling_params
        else:
            sampling_params_list = [sampling_params]*len(prompts)

        for prompt_data in prompts:
            if not isinstance(prompt_data, (str, list)):
                raise ValueError("Prompts must be a list of strings or a list of token ID lists.")
        return sampling_params_list

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

        if getattr(self, "_is_data_parallel_coordinator", False):
            return self._generate_data_parallel(prompts, sampling_params)

        sampling_params_list = self._normalize_generation_inputs(prompts, sampling_params)

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

    @staticmethod
    def _partition_prompt_indices(num_prompts: int, data_parallel_size: int) -> list[list[int]]:
        partitions = [[] for _ in range(data_parallel_size)]
        for idx in range(num_prompts):
            partitions[idx % data_parallel_size].append(idx)
        return partitions

    @staticmethod
    def _aggregate_data_parallel_metrics(per_rank_metrics: dict[int, dict], total_time: float) -> dict:
        total_tokens = sum(metrics.get("total_tokens", 0) for metrics in per_rank_metrics.values())
        metrics = {
            "total_tokens": total_tokens,
            "total_time": total_time,
            "throughput": total_tokens/total_time if total_time > 0 else 0,
            "per_rank": per_rank_metrics,
        }

        memory_values = [
            rank_metrics["avg_memory_mb"]
            for rank_metrics in per_rank_metrics.values()
            if "avg_memory_mb" in rank_metrics
        ]
        gpu_util_values = [
            rank_metrics["avg_gpu_util"]
            for rank_metrics in per_rank_metrics.values()
            if "avg_gpu_util" in rank_metrics
        ]
        if memory_values:
            metrics["avg_memory_mb"] = sum(memory_values)/len(memory_values)
        if gpu_util_values:
            metrics["avg_gpu_util"] = sum(gpu_util_values)/len(gpu_util_values)

        ttft_by_rank = {
            rank: rank_metrics["ttft"]
            for rank, rank_metrics in per_rank_metrics.items()
            if "ttft" in rank_metrics
        }
        tpot_by_rank = {
            rank: rank_metrics["tpot"]
            for rank, rank_metrics in per_rank_metrics.items()
            if "tpot" in rank_metrics
        }
        if ttft_by_rank:
            metrics["ttft"] = {"per_rank": ttft_by_rank}
        if tpot_by_rank:
            metrics["tpot"] = {"per_rank": tpot_by_rank}
        return metrics

    def _recv_data_parallel_message(self, rank: int, request_id: int | None):
        connection = self.dp_connections[rank]
        process = self.dp_processes[rank]
        while True:
            if connection.poll(0.1):
                message = connection.recv()
                if not isinstance(message, tuple) or not message:
                    raise RuntimeError(f"Malformed message from DP rank {rank}: {message!r}")
                if message[0] == "error":
                    if len(message) != 4:
                        raise RuntimeError(f"Malformed error message from DP rank {rank}: {message!r}")
                    _, error_request_id, error_rank, error_text = message
                    if request_id is None or error_request_id in (None, request_id):
                        raise RuntimeError(f"DP rank {error_rank} failed:\n{error_text}")
                return message
            if not process.is_alive():
                raise RuntimeError(f"DP rank {rank} exited unexpectedly.")

    def _generate_data_parallel(
        self,
        prompts: list[str],
        sampling_params: SamplingParams,
    ):
        sampling_params_list = self._normalize_generation_inputs(prompts, sampling_params)
        if not prompts:
            return [], self._aggregate_data_parallel_metrics({}, 0)

        partitions = self._partition_prompt_indices(len(prompts), self.config.data_parallel_size)
        request_id = self._dp_request_id
        self._dp_request_id += 1
        active_ranks = []
        start_time = time.time()

        for rank, indices in enumerate(partitions):
            if not indices:
                continue
            rank_prompts = [prompts[idx] for idx in indices]
            rank_sampling_params = [sampling_params_list[idx] for idx in indices]
            try:
                self.dp_connections[rank].send(
                    ("generate", request_id, indices, rank_prompts, rank_sampling_params)
                )
            except (BrokenPipeError, EOFError, OSError) as exc:
                raise RuntimeError(f"DP rank {rank} is not available.") from exc
            active_ranks.append(rank)

        outputs_by_index = {}
        per_rank_metrics = {}
        for rank in active_ranks:
            message = self._recv_data_parallel_message(rank, request_id=request_id)
            if len(message) != 6 or message[0] != "result" or message[1] != request_id:
                raise RuntimeError(f"Malformed result from DP rank {rank}: {message!r}")
            _, _, result_rank, indices, outputs, metrics = message
            if result_rank != rank:
                raise RuntimeError(f"DP rank {rank} returned result for rank {result_rank}.")
            if len(indices) != len(outputs):
                raise RuntimeError(
                    f"DP rank {rank} returned {len(outputs)} outputs for {len(indices)} prompts."
                )
            for prompt_idx, output in zip(indices, outputs):
                outputs_by_index[prompt_idx] = output
            per_rank_metrics[rank] = metrics

        if len(outputs_by_index) != len(prompts):
            missing = sorted(set(range(len(prompts)))-set(outputs_by_index))
            raise RuntimeError(f"DP generation did not return outputs for prompt indices {missing}.")

        total_time = time.time()-start_time
        outputs = [outputs_by_index[idx] for idx in range(len(prompts))]
        metrics = self._aggregate_data_parallel_metrics(per_rank_metrics, total_time)
        return outputs, metrics
