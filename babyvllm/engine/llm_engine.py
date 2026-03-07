import atexit
import time
from dataclasses import fields
import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer

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
        config = Config(model, **config_kwargs)
        
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
        
        # Construct scheduler after model runner, because scheduler needs to know the number of kv cache blocks.
        self.scheduler = Scheduler(config)
    
        # When the program exits, clean up resources automatically.
        atexit.register(self.exit)
        
    def exit(self):
        """ Clean up resources when the program exits. """
        
        self.model_runner.call('exit')
        del self.model_runner
        # Wait for all worker processes to finish.
        for process in self.processes:
            process.join()

    def add_request(self, prompt: str, sampling_params: SamplingParams):
        """ Add input prompt to the scheduler's waiting queue. """
        
        self.scheduler.add_sequence(Sequence(token_ids=self.tokenizer.encode(prompt), sampling_params=sampling_params))

    def step(self) -> tuple[list[int], int, bool]:
        """ Run the model for scheduled sequences. """
        
        # (1) Schedule sequences.
        scheduled_sequences, is_prefill = self.scheduler.schedule()
        # There is no sequence to schedule.
        if not scheduled_sequences:
            return [], is_prefill
        
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
        print_log: bool = False,
    ) -> list[str]:
        """ Generate text for all input prompts. """
        
        # Add all input prompts to the scheduler's waiting queue.
        for prompt in prompts:
            self.add_request(prompt, sampling_params)
        
        # {sequence id : generated tokens}
        generated_tokens = {}
        
        # Metrics collection
        total_tokens = 0
        total_time = 0
        memory_usages = []
        gpu_utilizations = []
        inference_start_time = time.time()
        
        while not self.scheduler.is_finished():
            start_time = time.time()
            # Call scheduler and model runner to run the model.
            # outputs: {sequence id : generated tokens}
            outputs, num_processed_tokens, is_prefill = self.step()
            end_time = time.time()
            
            # Collect metrics
            total_tokens += num_processed_tokens
            step_time = end_time-start_time
            total_time += step_time
            
            # Get memory usage
            memory_stats = torch.cuda.memory_stats()
            memory_usage = memory_stats['allocated_bytes.all.current']/(1024**2)  # Convert to MB
            memory_usages.append(memory_usage)
            
            # Get GPU utilization
            gpu_util = torch.cuda.utilization()
            gpu_utilizations.append(gpu_util)

            generated_tokens.update({seq_id : tokens for seq_id, tokens in outputs})

        # Sort generated tokens by sequence id. So, the output text are in the same order as user input.
        generated_tokens = [generated_tokens[seq_id] for seq_id in sorted(generated_tokens.keys())]
        
        # Calculate metrics
        inference_end_time = time.time()
        total_inference_time = inference_end_time-inference_start_time
        average_throughput = total_tokens/total_inference_time if total_inference_time > 0 else 0
        average_memory = sum(memory_usages)/len(memory_usages) if memory_usages else 0
        average_gpu_util = sum(gpu_utilizations)/len(gpu_utilizations) if gpu_utilizations else 0
        
        # Print metrics
        if print_log:
            print(f"\n=== Inference Metrics ===")
            print(f"Total tokens processed: {total_tokens}")
            print(f"Total inference time: {total_inference_time:.4f} seconds")
            print(f"Average throughput: {average_throughput:.2f} tokens/second")
            print(f"Average memory usage: {average_memory:.2f} MB")
            if gpu_utilizations:
                print(f"Average GPU utilization: {average_gpu_util:.2f}%")
            print("========================\n")
        
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in generated_tokens]
        return outputs
