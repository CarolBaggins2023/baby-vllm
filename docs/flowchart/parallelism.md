# Parallelism

## Source Modules

- `babyvllm/config.py`
- `babyvllm/engine/model_runner.py`
- `babyvllm/engine/llm_engine.py`
- `babyvllm/engine/async_llm_engine.py`
- `babyvllm/engine/sequence.py`
- `babyvllm/layers/linear.py`
- `babyvllm/layers/embedding_head.py`

BabyVllm has two parallel dimensions. Tensor parallelism splits one model replica across multiple local ranks. Data parallelism launches multiple full engine replicas and routes requests or prompt partitions across them.

## Tensor-Parallel Worker Protocol

```mermaid
sequenceDiagram
  participant Engine as "LLMEngine rank 0 side"
  participant Rank0 as "ModelRunner rank 0"
  participant SHM as "SharedMemory"
  participant Event as "Worker events"
  participant RankN as "ModelRunner rank > 0"

  Engine->>Rank0: "call('run', seqs)"
  Rank0->>SHM: "pickle run_worker_state + sequence worker states"
  Rank0->>Event: "set() for each worker"
  RankN->>Event: "wait()"
  RankN->>SHM: "read_shm()"
  RankN->>RankN: "run_worker_state(seq_states)"
  Rank0->>Rank0: "run(seqs)"
  RankN->>RankN: "run(seqs)"
  RankN->>Event: "clear()"
```

Rank 0 serializes compact worker states through shared memory. Decode states carry only the current token plus cache metadata; Prefill states carry full tokens because workers need absolute slices for the current chunk.

## Tensor-Parallel Layer Communication

```mermaid
flowchart TD
  Input["Replicated hidden states"] --> Column["ColumnParallelLinear family"]
  Column --> LocalShard["Each rank computes output feature shard"]
  LocalShard --> AttentionOrMLP["Local attention heads or MLP shard"]
  AttentionOrMLP --> Row["RowParallelLinear"]
  Row --> AllReduce["dist.all_reduce(sum)"]
  AllReduce --> Replicated["Replicated hidden states for next layer"]
  Replicated --> LMHead["ParallelLMHead"]
  LMHead --> Gather["dist.gather logits to rank 0"]
  Gather --> Sampling["Rank 0 sampling"]
```

`VocabParallelEmbedding` masks non-local token ids and all-reduces embeddings. `ParallelLMHead` gathers vocab shards to rank 0, where sampling happens.

## Offline Data Parallelism

```mermaid
flowchart TD
  Generate["LLMEngine.generate(prompts, sampling_params)"] --> Coordinator{"Coordinator with data_parallel_size > 1?"}
  Coordinator -->|No| Single["Single-replica offline generate"]
  Coordinator -->|Yes| Partition["_partition_prompt_indices(): round-robin prompt ids"]
  Partition --> Send["Send ('generate', request_id, indices, prompts, params) to each rank"]
  Send --> Worker["Offline DP worker owns full LLMEngine replica"]
  Worker --> LocalGenerate["worker llm.generate(rank_prompts, rank_params)"]
  LocalGenerate --> Result["Send ('result', request_id, rank, indices, outputs, metrics)"]
  Result --> Restore["Coordinator restores original prompt order"]
  Restore --> Metrics["_aggregate_data_parallel_metrics()"]
```

Offline DP partitions one batch by prompt index. Each worker blocks on its local `LLMEngine.generate()` call, then the coordinator merges outputs and metrics.

## Online Data Parallelism

```mermaid
sequenceDiagram
  participant API as "AsyncLLMEngine coordinator"
  participant Rank as "Online DP worker"
  participant Reader as "Coordinator reader task"
  participant Stream as "Coordinator AsyncStream"

  API->>API: "_select_data_parallel_rank() round-robin over live ranks"
  API->>Stream: "create AsyncStream(request_id)"
  API->>Rank: "send ('generate', request_id, prompt_token_ids, sampling_params)"
  Rank->>Rank: "async for output in local engine.generate()"
  Rank-->>Reader: "('output', request_id, rank, output)"
  Reader->>Stream: "stream.put(output)"
  Rank-->>Reader: "('done', request_id, rank)"
  Reader->>Stream: "stream.finish()"
```

Online DP routes each request to one full engine replica. Reader tasks keep worker Pipe messages flowing back into the coordinator's local streams. If a worker rank fails, the coordinator removes it from the live set and fails only the requests owned by that rank.

