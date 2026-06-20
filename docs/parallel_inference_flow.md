# BabyVllm End-to-End LLM Inference Flow and DP/TP Parallelism

This document is based on the current `baby-vllm` mainline code. It explains the complete path of a request from engine entry to final output, with a focus on how data parallelism (DP) and tensor parallelism (TP) are implemented for both offline and online inference.

Core code entry points:

| Topic | Main files |
| --- | --- |
| Configuration, device mapping, parallelism validation | `babyvllm/config.py` |
| Offline synchronous engine, offline DP coordinator | `babyvllm/engine/llm_engine.py` |
| Online asynchronous engine, online DP coordinator | `babyvllm/engine/async_llm_engine.py` |
| Scheduling, chunked prefill, KV block management | `babyvllm/engine/scheduler.py`, `babyvllm/engine/block_manager.py`, `babyvllm/engine/sequence.py` |
| TP workers, model execution, KV cache, CUDA Graph | `babyvllm/engine/model_runner.py` |
| Qwen3 TP layer implementation | `babyvllm/models/qwen3.py`, `babyvllm/layers/linear.py`, `babyvllm/layers/embedding_head.py`, `babyvllm/layers/attention.py` |
| OpenAI-compatible HTTP service | `babyvllm/entrypoints/cli.py`, `babyvllm/entrypoints/api_server.py` |
| Offline/online benchmarks | `offline_bench.py`, `online_bench.py` |

## 1. One-Sentence Overview

BabyVllm parallel inference is a two-level nested design:

- **DP, data parallelism**: the outer layer replicates complete model replicas. Each DP rank handles different requests or different prompt shards. DP ranks do not run model-forward collectives with each other. They mainly use parent-process `multiprocessing.Pipe` channels for task dispatch, result collection, and error propagation.
- **TP, tensor parallelism**: the inner layer shards a single model replica. One DP replica contains multiple TP ranks. Each rank owns a shard of weights, KV heads, and vocabulary, and all ranks cooperate during each forward pass through `torch.distributed` collectives.

Combined topology:

```text
DP coordinator process
|
+-- DP rank 0 process: single-replica engine
|   |
|   +-- TP rank 0 ModelRunner on cuda:0
|   +-- TP rank 1 worker process on cuda:1
|
+-- DP rank 1 process: single-replica engine
    |
    +-- TP rank 0 ModelRunner on cuda:2
    +-- TP rank 1 worker process on cuda:3
```

When `data_parallel_size = D` and `tensor_parallel_size = T`, the default setup needs `D * T` visible CUDA devices. The default physical device mapping is:

```text
physical_cuda_id = data_parallel_rank * tensor_parallel_size + tensor_parallel_rank
```

You can also use `data_parallel_device_ids` to explicitly specify the mapping from all logical ranks to physical GPUs.

## 2. First Distinguish DP and TP

| Dimension | DP | TP |
| --- | --- | --- |
| Parallel granularity | Request/prompt level | Inside a single model forward |
| Model weights | Every DP rank owns a complete replica | Each TP rank owns one shard of the current replica |
| KV cache | Each DP rank has an independent KV cache | Each TP rank allocates local KV cache for its local KV heads |
| Communication | Parent process and DP workers communicate through `multiprocessing.Pipe` | TP ranks cooperate through `torch.distributed` NCCL collectives and shared-memory control messages |
| Main benefit | Increases system throughput and isolates request queues | Reduces per-replica memory pressure and enables larger model forwards |
| Main cost | Multiple full model replicas consume multiple copies of memory | Each layer introduces all-reduce/gather communication overhead |
| Best fit | Many requests, model fits on one card | Model is tight on one card, or large-model decode/prefill needs multiple cards |

A common misconception is that DP splits one request across multiple GPUs. BabyVllm does not do that. A request belongs to exactly one DP rank. Only when that DP rank has `tensor_parallel_size > 1` will multiple GPUs cooperate inside that replica to execute the request's forward pass.

## 3. Configuration Initialization and Resource Isolation

`Config` is the source of all parallel behavior. It is responsible for:

1. Parsing and validating `tensor_parallel_size` and `data_parallel_size`.
2. Checking that `data_parallel_size * tensor_parallel_size` does not exceed the number of visible CUDA devices.
3. Generating an independent `distributed_init_method` for each DP replica.
4. Generating an independent `shared_memory_name` for each DP replica.
5. Validating Qwen3 TP shape constraints.

Key fields:

```python
tensor_parallel_size: int = 1
data_parallel_size: int = 1
data_parallel_rank: int = 0
data_parallel_world_size: int | None = None
data_parallel_base_port: int = 12345
data_parallel_device_ids: list[int] | None = None
distributed_init_method: str | None = None
shared_memory_name: str | None = None
```

### 3.1 How a DP Worker Configuration Becomes a Single Replica

When the parent process sees `data_parallel_size > 1`, it becomes a coordinator and does not load the model directly. For each DP rank, it calls `worker_config_kwargs(rank)`:

```text
coordinator config:
  data_parallel_size = D
  tensor_parallel_size = T

worker config:
  data_parallel_size = 1
  data_parallel_world_size = D
  data_parallel_rank = rank
  tensor_parallel_size = T
  distributed_init_method = None
  shared_memory_name = None
```

This means each DP worker internally sees itself as "a single-replica engine", while still knowing its own `data_parallel_rank`. That lets it obtain an independent device group, an independent TP init port, and an independent shared-memory name.

### 3.2 TP Shape Validation

The current Qwen3 TP implementation requires the following dimensions to be divisible by `tensor_parallel_size`:

- `num_attention_heads`
- `num_key_value_heads`
- `intermediate_size`

BabyVllm does not support KV-head replication, so if `num_key_value_heads < tensor_parallel_size` or the value is not divisible by TP size, it fails early. Vocabulary-parallel embedding pads the vocabulary until it is divisible by TP size, so the original vocab size itself does not need to be divisible.

## 4. Process Topology

### 4.1 DP=1, TP=1

```text
main process
|
+-- LLMEngine
    |
    +-- Scheduler
    +-- ModelRunner(rank=0)
    +-- Qwen3 model
```

There are no extra TP workers and no shared memory. The current `ModelRunner` still initializes a `torch.distributed` process group even when `world_size=1`, but the forward path does not produce cross-process collectives.

### 4.2 DP=1, TP=T

```text
main process
|
+-- LLMEngine
    |
    +-- Scheduler
    +-- ModelRunner(rank=0)
    |
    +-- spawn TP worker rank 1
    +-- spawn TP worker rank 2
    +-- ...
```

Rank 0 is the coordinator:

1. The parent engine directly calls only rank 0's `ModelRunner.call(...)`.
2. Rank 0 serializes the method name and arguments into shared memory.
3. Rank 0 sets multiprocessing `Event`s to notify rank 1..T-1.
4. All TP ranks call the same method, such as `run(...)`.
5. The forward pass cooperates internally through NCCL collectives.
6. Only rank 0 samples and returns token ids.

### 4.3 DP=D, TP=1

```text
parent process: offline or online DP coordinator
|
+-- DP rank 0 process: full model replica on cuda:0
+-- DP rank 1 process: full model replica on cuda:1
+-- ...
```

Each DP worker has a complete model and an independent scheduler/KV cache. The parent process does not run dense forward passes. It only dispatches requests and aggregates outputs.

### 4.4 DP=D, TP=T

```text
parent coordinator
|
+-- DP rank 0 process
|   |
|   +-- TP rank 0 on cuda:0
|   +-- TP rank 1 on cuda:1
|
+-- DP rank 1 process
    |
    +-- TP rank 0 on cuda:2
    +-- TP rank 1 on cuda:3
```

Important boundaries:

- There are no TP collectives across DP ranks.
- Each DP replica has its own `distributed_init_method`, preventing different replicas' TP groups from mixing.
- Each DP replica has its own shared-memory name, preventing TP control messages from crossing replica boundaries.
- Each DP worker starts only the TP workers inside its own replica.

## 5. Complete Offline Inference Flow

The offline entry point is `LLMEngine.generate(prompts, sampling_params)`. Offline inference means the caller provides a batch of prompts at once. The function blocks until the whole batch finishes, then returns `outputs, metrics`.

### 5.1 Offline DP Coordinator Initialization

When `LLMEngine(model, data_parallel_size=D, tensor_parallel_size=T)` and `D > 1`:

```text
LLMEngine.__init__
|
+-- Config(...)
+-- _is_data_parallel_coordinator = True
+-- load tokenizer only
+-- _init_data_parallel_workers()
    |
    +-- spawn process for DP rank 0
    +-- spawn process for DP rank 1
    +-- ...
    +-- wait for each worker to send ("ready", rank)
```

The parent process does not create a `ModelRunner`, load model weights, or allocate KV cache.

Each DP worker enters `data_parallel_offline_worker_process(...)`:

```text
worker process
|
+-- llm = LLMEngine(model, data_parallel_size=1, data_parallel_rank=rank, tensor_parallel_size=T)
+-- send ("ready", rank)
+-- loop:
    +-- receive ("generate", request_id, indices, prompts, sampling_params_list)
    +-- outputs, metrics = llm.generate(...)
    +-- send ("result", request_id, rank, indices, outputs, metrics)
```

Note that the `LLMEngine` created inside the worker is a single replica. It will still initialize TP workers according to `tensor_parallel_size=T`.

### 5.2 Offline Request Sharding

After the parent process receives `generate()`:

1. It validates `prompts` and `sampling_params`.
2. It assigns prompt indices to DP ranks by round-robin:

```python
rank = prompt_index % data_parallel_size
```

For example, with 7 prompts and DP=3:

```text
rank 0: prompt indices [0, 3, 6]
rank 1: prompt indices [1, 4]
rank 2: prompt indices [2, 5]
```

3. The parent sends each rank's prompt sub-list to the corresponding worker through a Pipe.
4. The parent waits for all active ranks to return.
5. It validates the count and indices returned by each rank.
6. It restores output order according to the original prompt indices.
7. It aggregates metrics.

This is DP as "data parallelism": different DP ranks compute different requests, and each request is assigned to only one rank.

### 5.3 Single-Replica Offline Execution

Whether it runs inside a DP worker or in a normal `data_parallel_size=1` offline engine, the path is:

```text
LLMEngine.generate
|
+-- normalize prompts and sampling params
+-- for each prompt:
|   +-- tokenize or accept token ids
|   +-- validate length and KV capacity
|   +-- create Sequence
|   +-- scheduler.add_sequence(seq)
|
+-- while scheduler is not finished:
    |
    +-- batch = scheduler.schedule()
    |
    +-- run_scheduled(batch.decode_sequences)
    |   +-- model_runner.call("run", decode_sequences)
    |   +-- scheduler.postprocess(...)
    |
    +-- run_scheduled(batch.prefill_sequences)
        +-- model_runner.call("run", prefill_sequences)
        +-- scheduler.postprocess(...)
```

`Scheduler.schedule()` returns a `ScheduledBatch` split into two physical sub-batches:

- `decode_sequences`
- `prefill_sequences`

This lets one logical step select both decode and prefill work, while the physical forward passes remain separated. A decode-only batch can use CUDA Graph replay. Prefill and chunked prefill use the eager path.

### 5.4 Offline Outputs and Metrics

`scheduler.postprocess(...)`:

1. Increments `seq.num_computed_tokens` according to `chunk_size`.
2. Drops the sampled token if the sequence is still inside chunked prefill, because the prompt has not been fully computed yet.
3. Appends the sampled token when prefill has completed or the sequence is already in decode.
4. Releases KV blocks and marks the sequence finished when EOS is reached, `max_tokens` is reached, or `max_model_length` is reached.

The final offline result decodes each sequence's generated token ids into:

```python
{"text": ..., "token_ids": ...}
```

The DP coordinator aggregates metrics as follows:

- `total_tokens`: summed over all active ranks.
- `total_time`: wall time from dispatch to all results collected by the parent.
- `throughput`: `total_tokens / total_time`.
- `per_rank`: preserves the original metrics from each rank.
- TTFT/TPOT percentiles are not synthesized into fake global values; the per-rank view is preserved.

## 6. Complete Online Inference Flow

The online entry point is either the OpenAI-compatible HTTP server or direct use of `AsyncLLMEngine.generate(...)`. Online inference means requests arrive independently, output is returned as token deltas in a stream, and each request has its own stream.

### 6.1 Service Startup

CLI path:

```text
babyvllm-server / python -m babyvllm.entrypoints.cli
|
+-- parse args
+-- build engine_kwargs
+-- AsyncLLMEngine(model, **engine_kwargs)
+-- api_server._engine = engine
+-- uvicorn.run(...)
```

The CLI exposes:

- `--tensor-parallel-size`
- `--data-parallel-size`
- scheduler/KV/cache/prefill parameters

The API server itself does not understand the internal details of DP/TP. It converts `/v1/completions` and `/v1/chat/completions` requests into `SamplingParams`, then calls `_engine.generate(...)`.

### 6.2 Online DP Coordinator Initialization

When `AsyncLLMEngine(model, data_parallel_size=D, tensor_parallel_size=T)` and `D > 1`:

```text
AsyncLLMEngine.__init__
|
+-- Config(...)
+-- engine = None
+-- tokenizer = AutoTokenizer
+-- _is_data_parallel_coordinator = True
+-- _init_data_parallel_workers()
    |
    +-- spawn online DP worker rank 0
    +-- spawn online DP worker rank 1
    +-- ...
    +-- wait for ("ready", rank)
```

The parent process does not create a local `LLMEngine`, load the model, or allocate KV cache.

Each online DP worker runs:

```text
data_parallel_worker_process
|
+-- asyncio.run(_data_parallel_worker_main)
    |
    +-- engine = AsyncLLMEngine(model, data_parallel_size=1, data_parallel_rank=rank, tensor_parallel_size=T)
    +-- send ("ready", rank)
    +-- loop receive command:
        |
        +-- ("generate", request_id, prompt_token_ids, sampling_params)
        |   +-- create asyncio task run_request(...)
        |
        +-- ("abort", request_id)
        |   +-- engine.abort(request_id)
        |
        +-- ("exit",)
            +-- stop worker
```

`run_request(...)` asynchronously iterates the local `engine.generate(...)` inside the worker. For each `RequestOutput`, it sends:

```text
("output", request_id, rank, output)
```

When the request completes, it sends:

```text
("done", request_id, rank)
```

### 6.3 Online Request Routing

The online DP coordinator's `generate(...)` uses `_add_data_parallel_request(...)`:

1. If the prompt is a string, tokenize it first with `tokenizer.encode`.
2. Allocate a `request_id`, or accept one provided by the caller.
3. Start DP reader tasks.
4. Select one live rank by round-robin.
5. Create mappings:

```text
request_id -> AsyncStream
request_id -> owner DP rank
request_id -> prompt_token_ids
request_id -> timing
```

6. Send through the Pipe:

```text
("generate", request_id, prompt_token_ids, sampling_params)
```

7. Return the `AsyncStream` to the caller.

The reader task continuously reads messages from DP workers:

```text
("output", request_id, rank, output)
  -> stream.put(output)

("done", request_id, rank)
  -> stream.finish()
  -> clean request mappings

("error", request_id, rank, error_text)
  -> fail only that request stream
```

If a rank as a whole fails, the coordinator removes it from the live-rank set and fails all requests owned by that rank.

### 6.4 Single-Replica Online Execution

When `data_parallel_size=1`, or when an online DP worker creates its local `AsyncLLMEngine`, the path is:

```text
AsyncLLMEngine.generate
|
+-- lazily start background engine loop
+-- add_request(...)
|   |
|   +-- tokenizer.encode if needed
|   +-- allocate request_id
|   +-- RequestTracker.add_request(...)
|   +-- return AsyncStream
|
+-- async for output in stream.generator():
    +-- yield RequestOutput
```

The background `_run_engine_loop()` repeatedly calls `_engine_step()`:

```text
_engine_step
|
+-- RequestTracker.get_new_and_aborted_requests()
+-- process aborts first
+-- land new requests:
|   +-- create Sequence
|   +-- engine.scheduler.add_sequence(seq)
|   +-- seq_id -> request_id
|   +-- request_id -> seq_id
|
+-- batch = engine.schedule()
|
+-- if decode sub-batch:
|   +-- engine.run_scheduled(decode_sequences)
|   +-- route RequestOutput immediately
|
+-- if prefill sub-batch:
    +-- engine.run_scheduled(prefill_sequences)
    +-- route RequestOutput
```

Why is decode routed first? Because an online service cares about streaming output and TTFT. Even if a logical step also contains a prefill sub-batch, sending decode tokens to the client before running later prefill work avoids blocking already-ready output behind prefill.

### 6.5 Online Output Format

`LLMEngine.step()` only produces:

```python
(seq_id, token_ids_delta, finished)
```

`AsyncLLMEngine` uses the `seq_id -> request_id` mapping to convert this into:

```python
RequestOutput(
    request_id=...,
    text=tokenizer.decode(token_ids_delta),
    token_ids=token_ids_delta,
    finished=...,
    prompt_token_ids=...,
    ttft=...,
    tpot=...,
    total_time=...,
)
```

The API server then converts `RequestOutput` into OpenAI-compatible JSON or SSE chunks.

### 6.6 Online Cancellation and Abort

Online cancellation has two layers:

1. The API layer catches `asyncio.CancelledError` and calls `_engine.abort(engine_request_id)`.
2. The engine layer releases the corresponding scheduler/KV resources.

For the online DP coordinator:

```text
abort(request_id)
|
+-- request_id -> owner rank
+-- finish local AsyncStream with CancelledError
+-- send ("abort", request_id) to owner DP worker
```

After the worker receives the abort command, it calls its local `engine.abort(request_id)`, and the local scheduler releases the request's KV blocks.

## 7. Core Scheduler, Sequence, and KV Cache Logic

### 7.1 Sequence Is the Scheduling Unit

`Sequence` stores the generation state of one request:

- `token_ids`: prompt tokens plus generated tokens.
- `num_prompt_tokens`: original prompt length.
- `num_computed_tokens`: number of tokens already forwarded and written into KV cache.
- `chunk_size`: amount of token compute assigned in the current step.
- `block_table`: mapping from logical blocks to physical KV blocks.
- Sampling parameters: temperature, max tokens, ignore EOS, max model length.

During chunked prefill, `num_computed_tokens` advances in segments. For a prompt length of 100 and chunk size 60:

```text
step 1: compute token [0, 60),  num_computed_tokens = 60,  no valid token is emitted
step 2: compute token [60,100), num_computed_tokens = 100, prefill completes and the first completion token can be produced
step 3: decode 1 token
step 4: decode 1 token
...
```

### 7.2 BlockManager Manages Physical KV Blocks

`BlockManager` maintains:

- `free_block_ids`
- `used_block_ids`
- `hash_to_block_id`
- each sequence's `block_table`

`allocate_chunk(seq, chunk_size)` computes which logical blocks are needed for `[num_computed_tokens, num_computed_tokens + chunk_size)`, then allocates or reuses physical blocks.

Complete blocks participate in the prefix-cache hash. Incomplete blocks are not inserted into `hash_to_block_id`, avoiding poor hit rates and collision risks from partial blocks.

### 7.3 Continuous Batching and Chunked Prefill

The scheduler has two queues:

- `waiting`: new requests that have not entered the running state.
- `running`: requests that already have KV blocks and are currently in prefill or decode.

Each `schedule()` call:

1. First selects decode sequences from `running`; each decode sequence currently needs only 1 token.
2. Uses the remaining sequence budget and token budget to select prefill sequences.
3. During prefill, continues already-running chunked prefills before admitting new waiting requests.
4. If decode work already exists, reserves KV blocks for future decode growth, reducing the chance that a sequence is preempted immediately after it starts decoding.

Return value:

```python
ScheduledBatch(
    decode_sequences=[...],
    prefill_sequences=[...],
)
```

The engine layer then executes these two physical batches separately.

### 7.4 Preemption

When KV blocks are insufficient, the scheduler may preempt some sequences:

1. Release the sequence's KV blocks.
2. Set `num_computed_tokens = 0`.
3. Put the sequence back at the front of `waiting`.

Recovery uses `seq.num_tokens`, not only `num_prompt_tokens`. If a sequence has already generated some tokens before being preempted, those generated tokens must also be prefilling context during recovery; otherwise, the model context would be lost.

## 8. ModelRunner and TP Worker Cooperation

`ModelRunner` bridges sequence state and model forward:

- It prepares input ids, positions, and attention metadata.
- It warms up the model and estimates available KV cache.
- It allocates KV cache.
- It captures the decode CUDA Graph.
- It calls Qwen3 forward and logits.
- It sends control messages between TP rank 0 and the other TP ranks.

### 8.1 TP Worker Startup

Single-replica `LLMEngine._init_single_replica()`:

```text
for rank in 1..T-1:
  spawn worker_process(config, rank, event)

rank 0:
  self.model_runner = ModelRunner(config, rank=0, event=events)
```

Each `ModelRunner`:

1. Sets the CUDA device based on `config.device_id_for_rank(rank)`.
2. Initializes the NCCL process group with `config.distributed_init_method`.
3. Creates the Qwen3 model.
4. Uses `load_model(...)` to load this rank's weight shards.
5. Warms up.
6. Allocates local KV cache.
7. Captures the decode CUDA Graph.
8. If `world_size > 1`, creates or attaches shared memory.

### 8.2 How Rank 0 Calls All TP Ranks

The parent engine calls only rank 0:

```python
outputs = self.model_runner.call("run", scheduled_sequences)
```

If `world_size > 1`, rank 0 does the following in `call(...)`:

1. Pickles the method name and arguments into shared memory.
2. Sets each worker's Event.
3. Directly calls the same method itself.

A TP worker's `loop()` does:

```text
while True:
  method_name, args = read_shm()
  call(method_name, *args)
  if method_name == "exit":
    break
```

This design uses rank 0 to send control-plane messages so every rank enters the same model method. The real data-plane tensor synchronization happens through NCCL collectives inside forward.

### 8.3 How Sequence State Is Sent to TP Workers

The argument to `run` is `list[Sequence]`. Pickling full objects directly would be fragile, so rank 0 calls:

```python
seq.to_worker_state()
```

The worker side reconstructs with:

```python
Sequence.from_worker_state(state)
```

The transported state is phase-aware:

- Prefill/chunked prefill: needs full `token_ids`, because `prepare_forward()` slices by absolute position.
- Decode: needs only compact state such as the current token, `num_computed_tokens`, `block_table`, and `chunk_size`; it does not need the full token history because the history is already in KV cache.

Shared memory is fixed at 1 MiB. If the payload exceeds capacity, rank 0 raises a clear error immediately, preventing workers from reading truncated data.

### 8.4 What prepare_forward Does

All TP ranks execute `prepare_forward(seqs)`. They get inputs with the same shape semantics, while each rank later uses its own local model shard.

It constructs:

- `input_ids`: tokens that should actually be forwarded in the current step.
- `positions`: absolute positions of these tokens.
- `cu_seqlens_q`: prefix sums for the current query chunks of each sequence.
- `cu_seqlens_k`: prefix sums for the visible context length of each sequence.
- `context_lens`: current visible length of each sequence.
- `block_tables`: physical KV blocks for each sequence.
- `slot_mapping`: physical slots where the current tokens should be written into KV cache.

It then uses `set_context(...)` to put attention metadata into a ContextVar. Attention and LM head read it through `get_context()` during forward.

### 8.5 run_model and Sampling

The logic of `run(seqs)` is:

```text
rank 0:
  temperatures = prepare_sample(seqs)
all ranks:
  input_ids, positions = prepare_forward(seqs)
  logits = run_model(input_ids, positions, is_decode_only)
rank 0:
  token_ids = sampler(logits, temperatures)
nonzero ranks:
  token_ids = None
all ranks:
  reset_context()
```

Only rank 0 samples. In TP mode, the LM head gathers vocab-sharded logits to rank 0. Nonzero ranks do not have complete logits and should not sample independently.

`run_model(...)` has two paths:

- Prefill or mixed dynamic shape: eager forward.
- Pure decode and not `enforce_eager`: CUDA Graph replay.

## 9. Qwen3 Tensor Parallel Implementation

### 9.1 Embedding: Vocabulary Parallelism

`VocabParallelEmbedding` shards along the vocabulary dimension:

```text
global vocab: [0 ................. vocab_size)

rank 0 owns [0,        partition)
rank 1 owns [partition, 2*partition)
...
```

In forward:

1. Each rank keeps only token ids owned by that rank.
2. Token ids not owned by the rank are masked to 0.
3. Local embedding lookup runs.
4. Masked outputs are set to 0.
5. `dist.all_reduce(SUM)` produces complete hidden states.

As a result, all TP ranks obtain identical hidden states for subsequent layers.

### 9.2 Attention: QKV Column Parallel + O Row Parallel

In Qwen3 attention:

```text
x replicated on all ranks
|
+-- QKVColumnParallelLinear
|   each rank computes local Q heads and local KV heads
|
+-- local attention
|   each rank computes only its own heads
|
+-- RowParallelLinear(o_proj)
    each rank computes a partial output
    all_reduce SUM -> every rank gets the full hidden_size output
```

`QKVColumnParallelLinear` slices the q/k/v checkpoint weights separately and packs them into one local qkv parameter. Each rank's local output dimension is:

```text
head_dim * (local_num_heads + 2 * local_num_kv_heads)
```

### 9.3 MLP: Gate/Up Column Parallel + Down Row Parallel

MLP path:

```text
x replicated
|
+-- MergedColumnParallelLinear(gate_proj + up_proj)
|   output features are sharded by TP rank
|
+-- SiluAndMul
|   local activation
|
+-- RowParallelLinear(down_proj)
    input features are sharded by TP rank
    all_reduce SUM -> full hidden_size output
```

`MergedColumnParallelLinear` uses `loaded_weight_id` to distinguish gate and up weights, slicing both checkpoint tensors into the same local merged parameter.

### 9.4 LM Head: Vocabulary-Parallel Logit Gather

`ParallelLMHead` reuses the vocabulary-parallel embedding weight sharding:

1. During prefill, it takes only the last query token's hidden state from each sequence to compute logits.
2. Each TP rank computes logits for its local vocab partition.
3. `dist.gather(..., dst=0)` gathers logits to rank 0.
4. Rank 0 concatenates logits and trims padded vocabulary.
5. Rank 0 samples.

The sampled token is not broadcast. Only rank 0 needs to return token ids to the scheduler. Other ranks will receive the sequence state needed for the next step from rank 0 through shared memory.

### 9.5 KV Cache: Local KV Heads per Rank

`ModelRunner.allocate_kv_cache()` allocates cache according to local KV heads:

```text
allocated_kv_cache shape:
  (2, num_layers, num_blocks, block_size, local_num_kv_heads, head_dim)
```

It then points each attention layer's `k_cache` and `v_cache` to the corresponding layer slice.

Attention forward:

1. If `slot_mapping` exists, use Triton `store_kvcache` to write current k/v into the local KV cache.
2. Prefill uses `flash_attn_varlen_func`.
3. Decode uses `flash_attn_with_kvcache`.

Because each TP rank owns only its own KV heads, the local KV cache also stores only those local KV heads.

## 10. How One Request Flows with Combined DP and TP

Take online `DP=2, TP=2` as an example:

```text
Client
|
+-- FastAPI /v1/completions
    |
    +-- AsyncLLMEngine coordinator
        |
        +-- choose owner DP rank by round-robin
        |
        +-- send request to DP rank 1
            |
            +-- DP rank 1 local AsyncLLMEngine
                |
                +-- local LLMEngine / Scheduler
                    |
                    +-- ModelRunner TP rank 0 on cuda:2
                    +-- TP worker rank 1 on cuda:3
                        |
                        +-- both ranks run Qwen3 forward
                        +-- row-parallel all_reduce
                        +-- lm_head gathers logits to local TP rank 0
                        +-- local TP rank 0 samples token
                    |
                    +-- scheduler.postprocess
                |
                +-- produce RequestOutput
            |
            +-- send ("output", request_id, dp_rank, output) to coordinator
        |
        +-- coordinator puts output into request AsyncStream
    |
    +-- FastAPI yields SSE chunk
```

This path shows:

- One request belongs to only one DP rank.
- That DP rank's TP ranks compute the request together.
- TP rank 0 is not global rank 0; it is the local rank 0 inside each DP replica.
- After completion, output returns along the same route to the coordinator and then to the HTTP client.

## 11. Key Differences Between Offline and Online

| Dimension | Offline `LLMEngine.generate` | Online `AsyncLLMEngine.generate` |
| --- | --- | --- |
| Request arrival | Caller passes the whole prompt batch at once | Requests arrive independently |
| Return style | Blocks until the whole batch completes | Async generator streams results |
| DP dispatch | Prompt indices are sharded by round-robin | Requests are routed round-robin to live DP ranks |
| Output ordering | Parent restores original prompt order | Each request has an independent AsyncStream |
| Cancellation | Batch failure raises an exception | Requests can be aborted independently and release KV cache |
| Metrics | Batch-level metrics and per-rank metrics | Request-level TTFT/TPOT/total_time; benchmark aggregates later |
| API layer | None | OpenAI-compatible completions/chat completions |

## 12. Error Handling and Cleanup

### 12.1 Initialization Failure

Early validation includes:

- DP/TP sizes must be positive integers.
- `tensor_parallel_size <= 8`.
- CUDA device count must cover `D * T`.
- Explicit device ids must be unique and within the visible-device range.
- Qwen3 TP dimensions must be valid.

After a DP coordinator starts workers, it waits for `("ready", rank)`. If a worker sends `("error", ...)` first or exits, initialization fails and already-started workers are cleaned up.

### 12.2 Offline Worker Failure

If an offline DP worker raises during `generate`, it sends:

```text
("error", request_id, rank, traceback)
```

The parent process raises `RuntimeError` after receiving it and does not return partial outputs.

### 12.3 Online Worker Failure

Online DP worker failures fall into two categories:

- One request fails: only that request stream fails.
- A rank disconnects or exits unexpectedly: the rank is removed from live ranks, all requests owned by it fail, and future requests are not routed to it.

### 12.4 cleanup

- `LLMEngine.exit()`: tells TP ranks to run `exit`, joins worker processes, and releases model/KV/cache/process group/shared memory resources.
- `AsyncLLMEngine.stop()`: stops the background loop or DP coordinator workers.
- During online DP coordinator shutdown, it first finishes all local streams, then sends `("exit",)` to workers and joins or terminates them.

These cleanup paths are idempotent in style. Repeated calls do not intentionally damage already-cleaned state.

## 13. How Benchmarks Cover DP/TP

### 13.1 Offline Benchmark

`offline_bench.py`:

1. Constructs a random-token workload.
2. Preflights CUDA device count for `dp_size * tp_size`.
3. Creates `LLMEngine(..., data_parallel_size=dp, tensor_parallel_size=tp)`.
4. Optionally runs warmup.
5. Calls `llm.generate(...)`.
6. Validates output count, metrics, and per-rank information.
7. Prints throughput and per-rank summaries.

### 13.2 Online Benchmark

`online_bench.py` supports:

- direct: directly calls `AsyncLLMEngine`.
- http: sends requests to an external or embedded API server.

In direct or embedded-server mode, `data_parallel_size`, `tensor_parallel_size`, and `data_parallel_device_ids` are passed into the engine. In external HTTP mode, the actual DP/TP settings are determined by the server startup command. The benchmark-side parallelism parameters are mainly used for reporting and consistency checks.

## 14. Practical Code Reading Order

If you want to understand the whole chain from source code, a good reading order is:

1. `Config.__post_init__()`: parallelism parameters, device mapping, Qwen3 TP validation.
2. `LLMEngine.__init__()`: distinguish offline DP coordinator from a single replica.
3. `LLMEngine.generate()`: complete offline batch lifecycle.
4. `Scheduler.schedule()` and `Scheduler.postprocess()`: continuous batching, chunked prefill, decode/prefill separation.
5. `ModelRunner.call()`, `write_shm()`, `loop()`: how TP rank 0 drives other TP ranks.
6. `ModelRunner.prepare_forward()`: attention metadata and KV slot generation.
7. `Qwen3Attention`, `Qwen3MLP`, `VocabParallelEmbedding`, `ParallelLMHead`: TP sharding and collectives.
8. `AsyncLLMEngine.__init__()`: distinguish online DP coordinator from a single replica.
9. `AsyncLLMEngine._engine_step()`: online request landing, scheduling, execution, and routing.
10. `api_server.py`: how HTTP request/response and SSE wrap `RequestOutput`.

## 15. Minimal Mental Model

You can think of BabyVllm as four layers:

```text
API / caller
  |
  | online: request stream
  | offline: prompt batch
  v
DP coordinator, optional
  |
  | chooses which full replica owns the work
  v
single replica engine
  |
  | scheduler chooses decode/prefill chunks and owns KV block lifecycle
  v
TP model runner group
  |
  | all TP ranks jointly run model forward
  | rank 0 samples and returns token ids
  v
outputs
```

DP solves the problem of distributing many requests across multiple full replicas. TP solves the problem of making one replica fit, or making one forward pass run across multiple GPUs. The scheduler solves the problem of deciding which sequences get KV blocks and token budget inside one replica at the current step. The online engine solves the problem of asynchronously landing multiple client requests in the scheduler and routing token deltas back to the correct stream.
