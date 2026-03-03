from transformers import AutoTokenizer
import time

from babyvllm.sampling_params import SamplingParams
from babyvllm.engine.llm_engine import LLMEngine

def main():
    model_name_or_path = '/root/autodl-tmp/Qwen/Qwen3-0.6B'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    llm = LLMEngine(
        model_name_or_path,
        enforce_eager=False,
        max_model_length=4096,
    )
    
    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=4096,
        max_model_length=4096,
    )
    
    prompts = [
        "Introduce yourself.",
        "List all prime numbers within 100.",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False,
            add_generation_prompt=True,
        ) for prompt in prompts
    ]
    
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()
    
    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")

    print(f"total time: {end_time-start_time}")

if __name__ == '__main__':
    main()
    