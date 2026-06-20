# BabyVllm Communication Methods

This document summarizes the main communication methods used in the current `baby-vllm` codebase, where they are implemented, what role they play, and why each boundary uses that method instead of other alternatives.

In this document, "communication" does not only mean network communication. It also covers in-process asynchronous queues, parent-child process IPC, Tensor Parallelism (TP) rank control messages, and GPU tensor collectives. The overall design in BabyVllm is: external service boundaries use HTTP, online request routing uses asyncio, Data Parallelism (DP) uses process-level messages, the TP control plane uses shared memory and events, and the TP data plane uses NCCL collectives.

## Overview

| Layer | Communication method | Main locations | Payload | Main role |
| --- | --- | --- | --- | --- |
| Client/server | HTTP JSON | `babyvllm/entrypoints/api_server.py`, `babyvllm/entrypoints/cli.py`, `online_bench.py` | OpenAI-compatible request bodies, non-streaming JSON responses | Expose `/v1/completions`, `/v1/chat/completions`, `/health`, and `/debug/stats` |
| Client/server | SSE over HTTP | `babyvllm/entrypoints/api_server.py` | `data: {...}\n\n` token deltas, `data: [DONE]` | Stream token deltas for online inference |
| In-process online routing | `asyncio.Queue` + `asyncio.Event` | `babyvllm/engine/request_tracker.py`, `babyvllm/engine/async_llm_engine.py` | New requests, aborted requests, per-request `RequestOutput` objects | Decouple API coroutines from the background engine loop |
| Offline DP | `torch.multiprocessing.Process` + `multiprocessing.Pipe` | `babyvllm/engine/llm_engine.py` | `ready/generate/result/error/exit` tuples | Parent process shards offline prompt batches across multiple full model replicas |
| Online DP | `torch.multiprocessing.Process` + `multiprocessing.Pipe` + asyncio reader tasks | `babyvllm/engine/async_llm_engine.py` | `ready/generate/output/done/error/abort/exit` tuples | Online coordinator routes each request to one DP worker and forwards streaming output back to the local stream |
| TP control plane | `multiprocessing.shared_memory.SharedMemory` + `multiprocessing.Event` | `babyvllm/engine/model_runner.py`, `babyvllm/engine/llm_engine.py` | Method names, worker sequence state, exit commands | TP rank 0 tells ranks greater than 0 to execute the same model runner method |
| TP data plane | `torch.distributed` NCCL collectives | `babyvllm/engine/model_runner.py`, `babyvllm/layers/linear.py`, `babyvllm/layers/embedding_head.py` | GPU tensors | All-reduce, gather, and barrier operations inside tensor-parallel model forward |
| TP rendezvous | `torch.distributed.init_process_group(init_method="tcp://...")` | `babyvllm/config.py`, `babyvllm/engine/model_runner.py` | Rank and world-size initialization data | Build an independent TP process group inside each DP replica |
| Model-layer metadata | `contextvars.ContextVar` | `babyvllm/utils/context.py`, `babyvllm/engine/model_runner.py`, `babyvllm/layers/attention.py` | Attention metadata, KV cache block tables, slot mappings | Pass current-forward metadata to attention and LM head code inside one process |

## Design Layers

BabyVllm's communication boundaries can be understood as the following stack:

```text
HTTP client / benchmark
  |
  | HTTP JSON or SSE
  v
FastAPI + Uvicorn
  |
  | async generator, AsyncStream, asyncio.Queue/Event
  v
AsyncLLMEngine
  |
  | optional online DP Pipe
  v
DP worker process, one full engine replica
  |
  | TP control: SharedMemory + Event
  | TP data: torch.distributed NCCL
  v
TP rank group inside one replica
```

Key points:

- DP is replica-level parallelism. A request belongs to exactly one DP rank, and DP ranks do not perform all-reduce during model forward.
- TP is model-internal parallelism. Multiple TP ranks inside one DP replica jointly execute one forward pass.
- The control plane and data plane are intentionally separated. Python objects, request lifecycle events, and method dispatch use lightweight control channels; large tensor synchronization uses NCCL.

## 1. Client and Server: HTTP JSON and SSE

Main code:

- `babyvllm/entrypoints/cli.py`
- `babyvllm/entrypoints/api_server.py`
- `online_bench.py`

The startup path is: `cli.py` creates an `AsyncLLMEngine`, injects it into `api_server._engine`, and then calls `uvicorn.run(api_server.app, host=..., port=...)`. The API layer exposes OpenAI-compatible endpoints:

- `POST /v1/completions`
- `POST /v1/chat/completions`
- `GET /health`
- `GET /v1/models`
- `GET /debug/stats`

Non-streaming requests use regular HTTP JSON. The API handler consumes all `RequestOutput` values from `_engine.generate(...)`, concatenates the full text, and returns a JSON response.

Streaming requests use SSE. `StreamingResponse` wraps an async generator. Each engine output is converted to a `data: <json>\n\n` line, and the stream ends with `data: [DONE]\n\n`.

Why HTTP and SSE are used here:

- HTTP JSON is the most common service boundary. `httpx`, curl, and OpenAI SDK-style clients can all call it easily.
- SSE fits LLM token streaming well. The server continuously pushes token deltas, while the client usually does not need to send control messages back on the same connection.
- SSE is simpler than WebSocket for this use case. It does not require extra bidirectional protocol state and is close to OpenAI-style streaming responses.
- gRPC is not used because the project is lightweight and learning-oriented. HTTP, Pydantic, and FastAPI are expressive enough for the current API surface and easier to debug.
- Clients are not connected directly to `AsyncLLMEngine` because cross-process, cross-machine, and benchmark modes all benefit from a stable network boundary.

`online_bench.py` contains two related modes:

- `direct`: calls `AsyncLLMEngine` directly, which removes HTTP overhead and helps observe engine behavior.
- `http`: uses `httpx.AsyncClient` to call either an external or embedded API server, which is closer to the real serving path.

## 2. In-Process Online Routing: AsyncStream, asyncio.Queue, asyncio.Event

Main code:

- `babyvllm/engine/request_tracker.py`
- `babyvllm/engine/async_llm_engine.py`

In online single-replica mode, a FastAPI handler calls:

```python
async for output in _engine.generate(prompt, sampling_params):
    ...
```

`AsyncLLMEngine.generate()` registers the request with `RequestTracker`. `RequestTracker` does three things:

- Creates an `AsyncStream` for each request.
- Puts the new request into `_new_requests: asyncio.Queue`.
- Sets `new_requests_event: asyncio.Event` to wake the background engine loop.

The background `_run_engine_loop()` waits on `new_requests_event` when idle. When requests arrive, it drains `_new_requests` and `_aborted_requests` and submits them to the scheduler. Each model step produces `RequestOutput` objects, and `RequestTracker.process_step_outputs()` routes those objects back to the corresponding `AsyncStream`. The API handler consumes `AsyncStream.generator()`, so every request has its own output queue.

Why asyncio primitives are used here:

- The API handler, background engine loop, and stream consumer all live in the same asyncio execution model, so `asyncio.Queue` and `asyncio.Event` are the natural synchronization tools.
- `asyncio.Event` lets an idle engine loop sleep instead of busy-polling.
- One `AsyncStream` per request preserves per-request FIFO ordering and prevents a slow client from blocking other request streams.
- `multiprocessing.Queue` is not used because this is not a cross-process boundary. Process IPC would add unnecessary pickling and context-switching cost.
- `threading.Event` and `queue.Queue` are not used because this path is not based on blocking threads. Blocking thread primitives would risk stalling the event loop.
- A simple callback is not enough because engine steps produce batched outputs across multiple requests and must also handle cancellation, exceptions, finish sentinels, and `aclose()` cleanup when a client disconnects.

Note that `asyncio.Queue` is a coroutine coordination primitive inside one event loop. It should not be treated as a cross-thread or cross-process communication channel.

## 3. Offline DP: Process + Pipe

Main code:

- `babyvllm/engine/llm_engine.py`
- `babyvllm/config.py`

When `LLMEngine(..., data_parallel_size=D)` is created with `D > 1`, the parent process becomes the offline DP coordinator and does not load the model. It creates one child process and one Pipe per DP rank:

```text
parent LLMEngine coordinator
  |
  +-- Pipe -> DP rank 0 process, full LLMEngine replica
  +-- Pipe -> DP rank 1 process, full LLMEngine replica
  +-- ...
```

The offline DP Pipe protocol is defined in `llm_engine.py`:

```text
worker -> coordinator: ("ready", rank)
coordinator -> worker: ("generate", request_id, indices, prompts, sampling_params_list)
worker -> coordinator: ("result", request_id, rank, indices, outputs, metrics)
worker -> coordinator: ("error", request_id | None, rank, error_text)
coordinator -> worker: ("exit",)
```

`_generate_data_parallel()` shards prompts by prompt index in a round-robin pattern:

```text
rank = prompt_index % data_parallel_size
```

Each worker runs a complete local `LLMEngine(data_parallel_size=1, data_parallel_rank=rank, tensor_parallel_size=T)`. After local generation finishes, the worker sends outputs and metrics back through its Pipe. The coordinator restores the original prompt order by prompt index.

Why offline DP uses Pipe:

- Offline DP is a parent-child request/response control flow: send a prompt batch and wait for each active rank to return a result. The bidirectional point-to-point model of `multiprocessing.Pipe` matches this shape well.
- Pipe automatically pickles Python objects, which is convenient for prompts, sampling parameters, metrics, and small to medium Python dict/list payloads.
- One Pipe per rank makes failures easy to localize. If a rank returns a malformed or error message, the coordinator can directly associate it with that rank.
- NCCL is not used because DP ranks own complete model replicas and process different requests. There is no GPU tensor reduction target across DP ranks.
- Shared memory is not used because offline DP sends task descriptions and final text/metrics, not large tensors that need zero-copy sharing.
- HTTP is not used because these workers are local internal execution replicas. They do not need exposed network services, port management, or a service protocol stack.
- A single global Queue is not used because the coordinator needs clear one-to-one lifecycle, ready/error/exit semantics, and request pairing for each rank.

## 4. Online DP: Pipe + asyncio Reader Task

Main code:

- `babyvllm/engine/async_llm_engine.py`

When `AsyncLLMEngine(..., data_parallel_size=D)` is created with `D > 1`, the parent process becomes the online DP coordinator. It also creates one full worker process and one Pipe per DP rank, but the protocol changes from "one batch returns one result" to "one request continuously returns outputs".

The online DP worker protocol is:

```text
worker -> coordinator: ("ready", rank)
coordinator -> worker: ("generate", request_id, prompt_token_ids, sampling_params)
worker -> coordinator: ("output", request_id, rank, RequestOutput)
worker -> coordinator: ("done", request_id, rank)
worker -> coordinator: ("error", request_id | None, rank, error_text)
coordinator -> worker: ("abort", request_id)
coordinator -> worker: ("exit",)
```

Request routing works as follows:

- `_select_data_parallel_rank()` performs round-robin selection over live ranks.
- `_add_data_parallel_request()` creates a local `AsyncStream` and records `request_id -> rank`.
- The coordinator sends `("generate", ...)` through that rank's Pipe.
- The worker creates an async task internally and calls its own local `AsyncLLMEngine.generate(...)`.
- Each time the worker produces a `RequestOutput`, it sends `("output", ...)`.
- The coordinator reader task receives the output and places it into the local `AsyncStream`, so FastAPI can continue producing SSE chunks from that stream.

Why online DP still uses Pipe:

- DP workers are local child processes, not standalone network services. Pipe is lighter than running an HTTP server inside each worker and avoids port allocation, health checking, and local load-balancing complexity.
- Each request belongs to only one DP rank, so there is no need for tensor collectives across DP ranks. Pipe only needs to carry request dispatch and output return messages.
- Streaming outputs are `RequestOutput` Python objects, which are small compared with token tensors. Pipe pickling is simple and sufficient for this data shape.
- One Pipe per rank preserves rank-level fault isolation. If a rank disconnects, the coordinator can remove it from `_live_data_parallel_ranks` and fail only the requests owned by that rank.
- Compared with a single shared Queue, point-to-point Pipes make `request_id -> owner rank` tracking and rank-level shutdown/error handling more direct.

Why asyncio reader tasks are needed:

- `multiprocessing.Connection.recv()` is a blocking API.
- An online server must not let a blocking `recv()` stall the FastAPI or engine event loop.
- Therefore, the coordinator uses `loop.run_in_executor(None, connection.recv)` inside `_data_parallel_reader_loop()`. The blocking wait runs in an executor, and message handling returns to the event loop.
- The worker side follows the same idea in `_data_parallel_worker_main()`: it waits for coordinator commands in an executor while local request tasks continue running asynchronously.

## 5. TP Control Plane: SharedMemory + Event

Main code:

- `babyvllm/engine/llm_engine.py`
- `babyvllm/engine/model_runner.py`
- `babyvllm/engine/sequence.py`

The TP worker process topology is:

```text
single replica engine process
  |
  +-- ModelRunner rank 0
  +-- TP worker process rank 1
  +-- TP worker process rank 2
  +-- ...
```

The engine only calls rank 0 directly:

```python
self.model_runner.call("run", scheduled_sequences)
```

When `tensor_parallel_size > 1`, rank 0 does the following in `ModelRunner.call()`:

1. Builds a worker message from the method name and arguments.
2. Writes `pickle.dumps(...)` output into shared memory.
3. Calls `set()` on each worker's `multiprocessing.Event`.
4. Executes the same method on rank 0 itself.

The rank greater than 0 worker loop:

1. Waits on its own Event.
2. Reads the payload length and payload from shared memory.
3. Uses `pickle.loads(...)` to recover the method name and arguments.
4. Clears the Event.
5. Executes the corresponding method, for example `run_worker_state(...)`.

The payload sent to TP workers is not the full `Sequence` object. It is a pickle-friendly state produced by `Sequence.to_worker_state()`:

- Prefill / chunked prefill needs full `token_ids`, because worker-side `prepare_forward()` slices by absolute position.
- Decode only needs compact state such as the current token, `num_computed_tokens`, `block_table`, and `chunk_size`, because historical tokens already live in the KV cache.

Why the TP control plane uses SharedMemory + Event:

- Rank 0 needs to broadcast the same control message to multiple local workers. Writing once to shared memory and letting all workers read the same payload is more direct than sending duplicate Pipe messages to each worker.
- Event only signals that a new command is available; shared memory carries the payload. Separating notification from data keeps the logic simple.
- Control messages are Python objects and sequence metadata, not GPU tensors. NCCL is not a good fit for this kind of object payload.
- `multiprocessing.Queue` is not used because rank 0 does not need workers to compete for tasks. All TP ranks must execute the same method, so queue work-stealing semantics do not match the requirement.
- Pipe is possible, but TP control is a broadcast from rank 0 to multiple workers. One Pipe per worker would duplicate serialization and add more connection management.
- `multiprocessing.Manager` is not used because it goes through a proxy process and IPC. That adds latency on a path where TP control messages are sent every step.

The current shared memory size is fixed at `2**20` bytes. `write_shm()` raises an error when the payload exceeds the capacity, which prevents workers from reading truncated data.

## 6. TP Data Plane: torch.distributed NCCL Collectives

Main code:

- `babyvllm/engine/model_runner.py`
- `babyvllm/layers/linear.py`
- `babyvllm/layers/embedding_head.py`
- `babyvllm/layers/attention.py`

Each TP rank calls `dist.init_process_group()` in `ModelRunner.__init__()`:

```python
dist.init_process_group(
    backend="nccl",
    init_method=config.distributed_init_method,
    world_size=config.tensor_parallel_size,
    rank=rank,
)
```

`Config._default_distributed_init_method()` generates a distinct local port for each DP rank:

```text
tcp://localhost:{data_parallel_base_port + data_parallel_rank}
```

As a result, with `DP=2, TP=2`, each DP replica has its own TP process group. TP collectives from different replicas do not mix with each other.

The main collectives during TP forward are:

- `VocabParallelEmbedding.forward()`: each rank looks up only its local vocabulary shard, masks tokens that do not belong to the rank, and then calls `dist.all_reduce(SUM)` to obtain the full embedding.
- `RowParallelLinear.forward()`: each rank computes a partial output and then calls `dist.all_reduce(SUM)` to obtain the full hidden states.
- `ParallelLMHead.forward()`: each rank computes local vocabulary logits, `dist.gather(..., dst=0)` collects them on rank 0, and rank 0 concatenates/trims them before sampling.
- `dist.barrier()` in `ModelRunner.__init__()`: synchronizes TP ranks around shared memory creation and connection timing.

Why the TP data plane uses NCCL:

- TP's core data is GPU tensors, and synchronization may happen in every layer. NCCL is PyTorch distributed's high-performance backend for GPU collectives.
- All-reduce and gather are part of the tensor-parallel math itself: row-parallel outputs must be summed, and vocabulary-parallel logits must be collected.
- These tensors should not go through Pipe, Queue, or Python shared memory. That would move GPU tensors through CPU/Python serialization paths, which are not suitable for per-layer forward latency and throughput.
- TP ranks must synchronize strictly within the same forward pass, and collective operations naturally provide that synchronization model.
- NCCL is not used for DP because DP ranks do not jointly compute the same request. The DP layer performs request-level dispatch, not tensor-level reduction.

One subtle point: after LM head gather, BabyVllm does not broadcast the sampled token to other ranks. Rank 0 is the only rank that returns tokens to the scheduler. On the next step, the sequence state needed by all TP workers is sent again by rank 0 through shared memory.

## 7. ContextVar: In-Process Model Metadata Channel

Main code:

- `babyvllm/utils/context.py`
- `babyvllm/engine/model_runner.py`
- `babyvllm/layers/attention.py`
- `babyvllm/layers/embedding_head.py`

`ModelRunner.prepare_forward()` builds attention metadata for the current physical forward:

- `cu_seqlens_q`
- `cu_seqlens_k`
- `slot_mapping`
- `block_tables`
- `context_lens`
- `is_prefill`

These values are stored in `contextvars.ContextVar` through `set_context(...)`. Attention layers and the LM head later read them through `get_context()`.

Why ContextVar is used here:

- This is an in-process function-call chain, so no IPC is needed.
- The attention layer forward signature does not need to carry a long list of scheduler metadata, which keeps the model modules closer to ordinary PyTorch modules.
- Compared with module-level globals, ContextVar is task-local. In online scenarios, different asyncio tasks do not overwrite each other's context.
- Compared with `threading.local`, ContextVar can distinguish different coroutines running on the same thread.
- Compared with `multiprocessing.Manager` or any other IPC mechanism, ContextVar is purely local and cheap enough for high-frequency forward-time reads.

Strictly speaking, ContextVar is not cross-process communication. It is an internal metadata-passing mechanism inside the model process. It is still worth documenting in the communication map because it determines how attention and KV cache code receive scheduler-layer information.

## 8. Why DP and TP Do Not Use the Same Communication Method

DP and TP both involve multiple GPUs, but their communication needs are fundamentally different.

| Dimension | DP | TP |
| --- | --- | --- |
| Parallelism granularity | Request / prompt level | Inside one model forward pass |
| Model weights | Each rank owns a full replica | Each rank owns weight shards |
| KV cache | Independent per DP rank | Each TP rank stores local KV heads |
| Main messages | Prompts, sampling parameters, `RequestOutput`, metrics | Hidden states, embeddings, logits, synchronization barriers |
| Communication frequency | Per request or per token output | Multiple times inside each layer forward |
| Data size and type | Mostly Python objects | Mostly GPU tensors |
| Suitable method | Pipe / asyncio stream | NCCL collectives plus local control messages |

Therefore:

- NCCL does not help DP, because there is no tensor reduction target across DP ranks.
- Pipe or SharedMemory alone is not enough for TP, because model shards need high-frequency GPU tensor communication.
- NCCL alone is not enough for TP either, because NCCL does not send Python method names, sequence state, or exit commands.
- Online request routing does not use TP shared memory because the online coordinator needs to manage request lifecycles, streams, aborts, and rank failures. It is not asking all workers to execute the same method.

## 9. End-to-End Path Examples

### Online, DP=1, TP=1

```text
client
  -> HTTP POST /v1/completions
  -> FastAPI handler
  -> AsyncLLMEngine.generate()
  -> RequestTracker._new_requests queue
  -> background engine loop
  -> Scheduler
  -> ModelRunner rank 0
  -> RequestTracker routes RequestOutput to AsyncStream
  -> SSE or JSON response
```

This path has no inter-process DP communication and no TP worker communication. Its main communication methods are HTTP/SSE and in-process asyncio queues/events.

### Online, DP=2, TP=2

```text
client
  -> HTTP/SSE
  -> AsyncLLMEngine DP coordinator
  -> Pipe: ("generate", request_id, prompt_token_ids, sampling_params)
  -> selected DP worker
  -> worker local AsyncLLMEngine + RequestTracker
  -> worker local ModelRunner TP rank 0
  -> SharedMemory + Event: notify TP rank 1
  -> NCCL all_reduce/gather inside model forward
  -> DP worker sends Pipe: ("output", request_id, rank, RequestOutput)
  -> coordinator reader task
  -> coordinator AsyncStream
  -> SSE chunk to client
```

This path uses all core communication methods. Each layer handles the data shape that best fits its boundary: the network layer handles HTTP, the DP layer handles request objects, the TP control plane handles methods and sequence state, and the TP data plane handles GPU tensors.

### Offline, DP=2, TP=1

```text
caller
  -> LLMEngine.generate(prompts, sampling_params)
  -> offline DP coordinator round-robin partitions prompt indices
  -> Pipe generate message to each active rank
  -> each worker runs local blocking LLMEngine.generate()
  -> Pipe result message returns outputs and metrics
  -> coordinator restores original prompt order
```

This path has no SSE and no online stream. Its communication model is a typical batch dispatch / gather flow.

## 10. Selection Principles

BabyVllm's communication design follows a few practical principles:

- Use standard protocols at external service boundaries: HTTP JSON/SSE makes client and benchmark integration straightforward.
- Use asyncio for in-process asynchronous boundaries: it avoids blocking the event loop and supports cancellation and streaming output.
- Use process-level messages for DP: DP ranks are full model replicas, so they only need to exchange requests and results.
- Separate the TP control plane from the TP data plane: Python control messages use SharedMemory/Event, while GPU tensor synchronization uses NCCL.
- Isolate the TP group of each DP replica: separate `distributed_init_method` values and `shared_memory_name` values prevent accidental communication across replicas.
- Do not force one communication method onto every layer: each layer chooses the tool that best matches its data shape, frequency, and synchronization semantics.

For future communication-layer extensions, first identify which boundary the new requirement belongs to:

- External users or cross-machine communication: prefer service protocols such as HTTP or gRPC.
- Coroutines inside one process: prefer asyncio Queue/Event/Task.
- Small Python objects between local parent and child processes: use Pipe or Queue.
- Broadcasting the same local control message to multiple workers: use SharedMemory + Event.
- GPU tensor collectives: use `torch.distributed` / NCCL.

This separation keeps request routing, lifecycle management, and model-layer tensor synchronization from being mixed into one oversized communication mechanism.
