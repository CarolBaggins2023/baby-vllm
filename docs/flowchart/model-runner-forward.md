# Model Runner Forward

## Source Modules

- `babyvllm/engine/model_runner.py`
- `babyvllm/engine/sequence.py`
- `babyvllm/utils/context.py`
- `babyvllm/layers/sampler.py`
- `babyvllm/models/qwen3.py`

`ModelRunner` converts scheduled `Sequence` objects into tensors and attention metadata, runs Qwen3, computes logits, samples token ids, and resets the task-local context after every physical forward.

## Run Pipeline

```mermaid
flowchart TD
  Start["ModelRunner.call('run', seqs)"] --> TP{"tensor_parallel_size > 1 and rank == 0?"}
  TP -->|Yes| WriteSHM["write_shm(): serialize method and worker sequence states"]
  WriteSHM --> Wake["Set worker events"]
  TP -->|No| Local
  Wake --> Local["run(seqs) on local rank"]
  Local --> SamplePrep["Rank 0 prepare_sample(): temperatures"]
  SamplePrep --> ForwardPrep["prepare_forward(seqs)"]
  ForwardPrep --> SetContext["set_context(attention metadata)"]
  SetContext --> RunModel["run_model(input_ids, positions, is_decode_only)"]
  RunModel --> Logits["Qwen3ForCausalLM + compute_logits()"]
  Logits --> Sample["Rank 0 sampler(logits, temperatures)"]
  Sample --> Reset["reset_context()"]
  Reset --> Return["Return sampled token ids to scheduler"]
```

Worker ranks execute the same `run()` method after rank 0 publishes a compact worker state. Decode worker states only include the current token and KV block table; Prefill states include full token ids so `prepare_forward()` can slice by absolute token index.

## `prepare_forward()` Metadata

```mermaid
flowchart TD
  Start["prepare_forward(seqs)"] --> Init["Initialize input_ids, positions, seqlens, block_tables, slot_mapping"]
  Init --> Loop["For each scheduled sequence"]
  Loop --> Range["start=num_computed_tokens, end=start+chunk_size"]
  Range --> Tokens["Append token_ids[start:end] and positions[start:end]"]
  Tokens --> Lens["q_len=chunk_size, k_len=end"]
  Lens --> Prefix["Update cu_seqlens_q and cu_seqlens_k"]
  Prefix --> ContextLens["Append context_lens=end"]
  ContextLens --> Slots["Map each logical token to physical block slot"]
  Slots --> Next{"More sequences?"}
  Next -->|Yes| Loop
  Next -->|No| Tables["Pad seq.block_table rows into block_tables tensor"]
  Tables --> DecodeOnly{"All seqs are one-token Decode?"}
  DecodeOnly -->|Yes| DecodeContext["set_context(is_prefill=False)"]
  DecodeOnly -->|No| PrefillContext["set_context(is_prefill=True)"]
  DecodeContext --> Return["Return input_ids, positions"]
  PrefillContext --> Return
```

The most important tensors are:

- `slot_mapping`: where newly computed K/V values are written in the physical KV cache.
- `block_tables`: which physical blocks contain each sequence history.
- `cu_seqlens_q`: current query chunk boundaries.
- `cu_seqlens_k`: visible context boundaries, including cached history plus the current chunk.
- `context_lens`: per-sequence visible context length for decode-with-cache.

## Eager Prefill And CUDA Graph Decode

```mermaid
flowchart TD
  RunModel["run_model(input_ids, positions, is_decode_only)"] --> Branch{"is_decode_only and not enforce_eager?"}
  Branch -->|No| Eager["Eager path: model(input_ids, positions)"]
  Eager --> EagerLogits["model.compute_logits(hidden_states)"]
  Branch -->|Yes| GraphPick["Pick smallest captured graph batch size >= current bs"]
  GraphPick --> Copy["Copy input_ids, positions, slot_mapping, context_lens, block_tables"]
  Copy --> Replay["graph.replay()"]
  Replay --> GraphLogits["compute_logits(graph_vars.outputs[:bs])"]
  EagerLogits --> Return["Return logits"]
  GraphLogits --> Return
```

Prefill and chunked Prefill are eager because their tensor shapes are dynamic. Pure Decode is shape-stable enough to replay a captured CUDA graph, which avoids kernel-launch overhead for the common one-token-per-sequence path.

