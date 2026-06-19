# Online Streaming

## Source Modules

- `babyvllm/engine/async_llm_engine.py`
- `babyvllm/engine/request_tracker.py`
- `babyvllm/engine/outputs.py`
- `babyvllm/entrypoints/api_server.py`

Online serving keeps request admission, scheduling, execution, and output routing inside one async engine loop. API handlers consume per-request streams while the background loop batches work across all live requests.

```mermaid
sequenceDiagram
  participant Client as "HTTP client"
  participant API as "FastAPI handler"
  participant AsyncEngine as "AsyncLLMEngine"
  participant Tracker as "RequestTracker"
  participant EngineLoop as "_run_engine_loop()"
  participant Engine as "LLMEngine"
  participant Stream as "AsyncStream"

  Client->>API: "POST /v1/completions or /v1/chat/completions"
  API->>AsyncEngine: "generate(prompt, sampling_params)"
  AsyncEngine->>AsyncEngine: "lazy start background loop if needed"
  AsyncEngine->>Tracker: "add_request(request_id, prompt_token_ids)"
  Tracker-->>AsyncEngine: "AsyncStream"
  Tracker->>EngineLoop: "new_requests_event.set()"
  EngineLoop->>Tracker: "get_new_and_aborted_requests()"
  EngineLoop->>Engine: "add Sequence to scheduler"
  EngineLoop->>Engine: "schedule()"
  EngineLoop->>Engine: "run_scheduled(decode_sequences)"
  Engine-->>EngineLoop: "seq token deltas"
  EngineLoop->>Tracker: "process_step_outputs(RequestOutput)"
  Tracker->>Stream: "put(output)"
  Stream-->>API: "async iterator yields output"
  API-->>Client: "JSON chunk or SSE data line"
```

## Engine Loop

```mermaid
flowchart TD
  LoopStart["_engine_step()"] --> Drain["Drain aborted requests and new requests"]
  Drain --> AbortFirst["Abort scheduler sequences first"]
  AbortFirst --> Land["Create Sequence objects for new requests"]
  Land --> HasWork{"New work or scheduler not finished?"}
  HasWork -->|No| Idle["wait_for_new_requests()"]
  HasWork -->|Yes| Schedule["engine.schedule()"]
  Schedule --> Decode{"Decode sub-batch?"}
  Decode -->|Yes| RunDecode["run_scheduled(decode_sequences)"]
  RunDecode --> RouteDecode["_route_engine_outputs()"]
  RouteDecode --> Yield["asyncio.sleep(0)"]
  Decode -->|No| Prefill
  Yield --> Prefill{"Prefill sub-batch?"}
  Prefill -->|Yes| RunPrefill["run_scheduled(prefill_sequences)"]
  RunPrefill --> RoutePrefill["_route_engine_outputs()"]
  Prefill -->|No| Done["Return had_work = True"]
  RoutePrefill --> Done
  Idle --> LoopStart
  Done --> LoopStart
```

## Abort Path

```mermaid
flowchart TD
  Disconnect["Client disconnects or caller cancels generator"] --> AClose["AsyncStream iterator aclose()"]
  AClose --> TrackerAbort["RequestTracker.abort_request()"]
  TrackerAbort --> AbortQueue["Queue request_id in _aborted_requests"]
  TrackerAbort --> FinishStream["Finish stream with CancelledError"]
  AbortQueue --> EngineStep["Next _engine_step()"]
  EngineStep --> Map["request_id -> seq_id"]
  Map --> SchedulerAbort["scheduler.abort_sequence(seq_id)"]
  SchedulerAbort --> FreeKV["BlockManager.deallocate()"]
  FreeKV --> Cleanup["Remove seq/request/prompt/timing mappings"]
```

`RequestTracker` keeps the async stream lifecycle separate from scheduler state. That makes cancellation idempotent: it can finish the stream immediately, then let the next engine iteration free KV blocks and remove scheduler entries.

