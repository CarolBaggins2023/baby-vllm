# Qwen3 Model

## Source Modules

- `babyvllm/models/qwen3.py`
- `babyvllm/layers/embedding_head.py`
- `babyvllm/layers/linear.py`
- `babyvllm/layers/attention.py`
- `babyvllm/layers/rotary_embedding.py`
- `babyvllm/layers/layernorm.py`
- `babyvllm/layers/activation.py`
- `babyvllm/layers/sampler.py`

BabyVllm implements Qwen3 as a compact causal LM stack. The model returns hidden states from `forward()`, and `ModelRunner.run_model()` calls `compute_logits()` afterward so CUDA graph replay can reuse the model forward output buffer.

## Causal LM Path

```mermaid
flowchart TD
  Input["input_ids + positions"] --> Model["Qwen3ForCausalLM.forward()"]
  Model --> Embed["Qwen3Model.embed_tokens"]
  Embed --> Layers["Repeated Qwen3DecoderLayer"]
  Layers --> FinalNorm["Final RMSNorm"]
  FinalNorm --> Hidden["Hidden states"]
  Hidden --> LMHead["compute_logits(): ParallelLMHead"]
  LMHead --> Logits["Logits on rank 0 after TP gather"]
  Logits --> Sampler["Sampler: temperature + softmax + exponential race"]
  Sampler --> Token["Sampled token ids"]
```

For Prefill, `ParallelLMHead` uses `context.cu_seqlens_q[1:] - 1` to select the last token of each scheduled chunk before computing logits. Decode already has one query token per sequence.

## Decoder Layer

```mermaid
flowchart TD
  X["x, residual"] --> InputNorm["input_layernorm"]
  InputNorm --> Attention["Qwen3Attention"]
  Attention --> PostNorm["post_attention_layernorm"]
  PostNorm --> MLP["Qwen3MLP"]
  MLP --> Out["x, residual"]
```

## Attention Block

```mermaid
flowchart TD
  X["Hidden states"] --> QKV["QKVColumnParallelLinear"]
  QKV --> Split["Split q, k, v"]
  Split --> Heads["View local heads"]
  Heads --> Norm{"attention_bias is False?"}
  Norm -->|Yes| QKNorm["RMSNorm on q and k"]
  Norm -->|No| Rotary
  QKNorm --> Rotary["RotaryEmbedding(positions, q, k)"]
  Rotary --> Flash["Attention(q, k, v)"]
  Flash --> Merge["Flatten local heads"]
  Merge --> OProj["RowParallelLinear o_proj"]
  OProj --> Output["Attention output"]
```

`QKVColumnParallelLinear` shards Q/K/V output features by tensor-parallel rank. `RowParallelLinear` all-reduces the projected attention output so the next block receives replicated hidden states.

## MLP Block

```mermaid
flowchart TD
  X["Hidden states"] --> GateUp["MergedColumnParallelLinear: gate + up"]
  GateUp --> Activation["SiluAndMul"]
  Activation --> Down["RowParallelLinear down_proj"]
  Down --> Output["MLP output"]
```

The gate and up projections are packed into one column-parallel layer. The down projection is row-parallel and reduces across ranks.

