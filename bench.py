import os
import time
import numpy as np
from random import randint, seed
from transformers import AutoTokenizer
import torch
import torch.profiler as profiler

from babyvllm import SamplingParams
from babyvllm.engine.llm_engine import LLMEngine

def llm_generate():
    # 1. Benchmark basic parameters setting
    seed(42)  # Fix random seed for reproducibility
    num_seqs = 256
    max_input_len = 1024
    max_output_len = 1024

    path = '/root/autodl-tmp/Qwen/Qwen3-0.6B'
    
    print("Initializing baby-vllm engine...")
    # Configure enough max_num_batched_tokens, and enable enforce_eager=False to enable CUDA Graph
    llm = LLMEngine(
        model=path, 
        enforce_eager=False, 
        tensor_parallel_size=1,
        max_num_batched_tokens=2048,
        max_num_sequences=256
    )

    # 2. Construct random token ID data (completely bypass Tokenizer)
    print(f"Generating {num_seqs} random requests with varying lengths...")
    # Simulate input sequences (length between 100 and 1024)
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))] 
        for _ in range(num_seqs)
    ]
    # Simulate output constraints (force ignore EOS, generate length between 100 and 1024)
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_output_len)) 
        for _ in range(num_seqs)
    ]

    # 3. Warmup system (Warmup)
    print("Engine warming up (compile CUDA Graph, allocate memory)...")
    llm.generate(["Benchmark Warmup!"], SamplingParams(max_tokens=8))
    
    # 4. Benchmark start
    print("\n" + "="*40)
    print("Start Benchmark (Continuous Batching + Chunked Prefill) ")
    print("="*40)
    
    # Input the generated token_ids list
    outputs, metrics = llm.generate(prompt_token_ids, sampling_params)
    
    # Print metrics
    print("\n" + "="*50)
    print("baby-vllm Benchmark Results")
    print("="*50)
    print(f"Total Requests            : {num_seqs} sequences")
    print(f"Total Output Tokens       : {metrics['total_tokens']} tokens")
    print(f"JCT (Job Completion Time) : {metrics['total_time']:.2f} s")
    print(f"System Throughput         : {metrics['throughput']:.2f} tokens/s")
    
    print(f"\n--- TTFT (Time To First Token) ---")
    print(f"Avg: {metrics['ttft']['avg']:.4f} s")
    print(f"P50: {metrics['ttft']['p50']:.4f} s")
    print(f"P90: {metrics['ttft']['p90']:.4f} s")
    print(f"P99: {metrics['ttft']['p99']:.4f} s")
    
    print(f"\n--- TPOT (Time Per Output Token) ---")
    print(f"Avg: {metrics['tpot']['avg']:.4f} s")
    print(f"P50: {metrics['tpot']['p50']:.4f} s")
    print(f"P90: {metrics['tpot']['p90']:.4f} s")
    print(f"P99: {metrics['tpot']['p99']:.4f} s")
    print("="*50)

def export_trace():
    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        llm_generate()
    
    prof.export_chrome_trace("trace_baby.json")

def main():
    llm_generate()
    # export_trace()

if __name__ == "__main__":
    main()