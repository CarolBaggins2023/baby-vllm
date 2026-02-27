import os
from transformers import AutoTokenizer, AutoConfig

from babyvllm.sampling_params import SamplingParams
from babyvllm.engine.llm_engine import LLMEngine

def main():
    model_name_or_path = '/root/autodl-tmp/Qwen/Qwen3-0.6B'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    config = {
        'model_name_or_path': model_name_or_path,
        'max_num_sequences': 16,
        'max_num_batched_tokens': 16384,
        'max_cached_blocks': 1024,
        'block_size': 256,
        'world_size': 1,
        'enforce_eager': True,
        'vocab_size': max(tokenizer.vocab_size, model_config.vocab_size),
        'hidden_size': model_config.hidden_size,
        'num_heads': model_config.num_attention_heads,
        'head_dim': model_config.hidden_size // model_config.num_attention_heads,
        'num_kv_heads': model_config.num_key_value_heads,
        'intermediate_size': model_config.intermediate_size,
        'num_layers': model_config.num_hidden_layers,
        'tie_word_embeddings': model_config.tie_word_embeddings if hasattr(model_config, 'tie_word_embeddings') else True,
        'base': 10000,
        'rms_norm_epsilon': model_config.rms_norm_eps,
        'qkv_bias': False,
        'scale': 1,
        'max_position': model_config.max_position_embeddings,
        'ffn_bias': True,
        'max_model_length': min(4096, model_config.max_position_embeddings),
        'gpu_memory_utilization': 0.9,
        'eos': tokenizer.eos_token_id if tokenizer.eos_token_id is not None else model_config.eos_token_id,
    }
    
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
            [{'role': 'user', 'content': prompt}],
            tokenize=False,
            add_generation_prompt=True,
        ) for prompt in prompts
    ]
    
    outputs = llm.generate(prompts, sampling_params)
    
    for prompt, output in zip(prompts, outputs['text']):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output!r}")

if __name__ == '__main__':
    main()
    