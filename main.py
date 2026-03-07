import time
import torch
import torch.profiler as profiler
from transformers import AutoTokenizer

from babyvllm.sampling_params import SamplingParams
from babyvllm.engine.llm_engine import LLMEngine


def PrintCompletionOutputs(prompts, outputs):
    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")

def LLMGenerate():
    model_name_or_path = '/root/autodl-tmp/Qwen/Qwen3-0.6B'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    llm = LLMEngine(model_name_or_path, enforce_eager=False, tensor_parallel_size=1)
    
    sampling_params = SamplingParams(temperature=0.6, max_tokens=512)
    
    prompts = [
        "Introduce yourself.",
        "List all prime numbers within 100.",
    ]*10
    prompts = [
        tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False,
            add_generation_prompt=True,
        ) for prompt in prompts
    ]
    
    outputs = llm.generate(prompts, sampling_params, print_log=True)
    
    # PrintCompletionOutputs(prompts, outputs)

def ExportTrace():
    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        LLMGenerate()
    
    prof.export_chrome_trace("trace_baby.json")

def main():
    LLMGenerate()
    # ExportTrace()

if __name__ == '__main__':
    main()
    