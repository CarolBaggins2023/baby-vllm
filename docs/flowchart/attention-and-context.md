# Attention And Context

## Source Modules

- `babyvllm/utils/context.py`
- `babyvllm/layers/attention.py`
- `babyvllm/engine/model_runner.py`
- `babyvllm/models/qwen3.py`

BabyVllm stores attention metadata in a `contextvars.ContextVar`. The model layers can call `get_context()` without receiving many metadata tensors through every forward signature, while concurrent asyncio tasks keep isolated context values.

## Context Lifecycle

```mermaid
sequenceDiagram
  participant Runner as "ModelRunner"
  participant ContextVar as "ContextVar"
  participant Qwen3 as "Qwen3 model"
  participant Attention as "Attention layer"

  Runner->>Runner: "prepare_forward(seqs)"
  Runner->>ContextVar: "set_context(metadata)"
  Runner->>Qwen3: "model(input_ids, positions)"
  Qwen3->>Attention: "attention(q, k, v)"
  Attention->>ContextVar: "get_context()"
  ContextVar-->>Attention: "Context"
  Attention-->>Qwen3: "attention output"
  Qwen3-->>Runner: "hidden states"
  Runner->>ContextVar: "reset_context()"
```

`Context` carries `is_prefill`, cumulative sequence lengths, max sequence lengths, `slot_mapping`, `block_tables`, and `context_lens`.

## KV Cache Write

```mermaid
flowchart TD
  Attn["Attention.forward(q, k, v)"] --> Context["context = get_context()"]
  Context --> HasCache{"k_cache/v_cache allocated and slot_mapping exists?"}
  HasCache -->|Yes| Store["store_kvcache(k, v, k_cache, v_cache, slot_mapping)"]
  Store --> Kernel["Triton kernel writes one token per program"]
  Kernel --> Path{"context.is_prefill?"}
  HasCache -->|No| Path
```

`slot_mapping` is indexed by current input token. Each value is an absolute slot inside the preallocated KV cache, computed from `physical_block_id * block_size + block_offset`.

## Prefill And Decode Attention Paths

```mermaid
flowchart TD
  Path{"context.is_prefill?"} -->|Yes| Prefill["Prefill or chunked Prefill"]
  Prefill --> HasBT{"block_tables is not None?"}
  HasBT -->|Yes| UseCache["Use k_cache/v_cache with block_table"]
  HasBT -->|No| DirectKV["Use current k/v directly"]
  UseCache --> Varlen["flash_attn_varlen_func(cu_seqlens_q, cu_seqlens_k, block_table)"]
  DirectKV --> Varlen
  Path -->|No| Decode["Decode"]
  Decode --> CacheAttn["flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache, context_lens, block_table)"]
  Varlen --> Output["Return attention output"]
  CacheAttn --> Output
```

Prefill uses `flash_attn_varlen_func` because each sequence can contribute a different query chunk length. Decode uses `flash_attn_with_kvcache` because each sequence contributes one new query token and attends to historical K/V through the cache.

