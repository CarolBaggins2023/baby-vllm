import os
from transformers import AutoTokenizer

from babyvllm.sampling_params import SamplingParams
from babyvllm.engine.llm_engine import LLMEngine

config = {
    'max_num_sequences': 16,
    'max_num_batched_tokens': 1024,
    'max_cached_blocks': 1024,
    'block_size': 256,
    'world_size': 1,
    'model_name_or_path': 'gpt2',
    'enforce_eager': True,
    'vocab_size': 151643,
    'hidden_size': 1024,
    'num_heads': 16,
    'head_dim': 64,
    'num_kv_heads': 8,
    'intermediate_size': 3072,
    'num_layers': 28,
    'tie_word_embeddings': True,
    'base': 10000,
    'rms_norm_epsilon': 1e-6,
    'qkv_bias': False,
    'scale': 1,
    'max_position': 128, # should be >= max_model_length, max position index allowed in rotary embedding
    'ffn_bias': True,
    'max_num_batch_tokens': 4096,
    'max_model_length': 128,
    'gpu_memory_utilization': 0.9,
    'eos': 151642,
}

def main():
    model_path = "/root/autodl-tmp/Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLMEngine(config=config)
    
    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=256,
        max_model_length=128
    )
    
    prompts = [
        "Introduce yourself.",
        "List all prime numbers within 100.",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{'role': 'user', 'context': prompt}],
            tokenize=False,
            add_generation_prompt=True,
        ) for prompt in prompts
    ]
    
    outputs = llm.generate(prompts, sampling_params)
    
    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")

if __name__ == '__main__':
    main()
    