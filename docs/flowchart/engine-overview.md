# Engine Overview

## Source Modules

- `babyvllm/entrypoints/cli.py`
- `babyvllm/entrypoints/api_server.py`
- `babyvllm/engine/llm_engine.py`
- `babyvllm/engine/async_llm_engine.py`
- `babyvllm/engine/scheduler.py`
- `babyvllm/engine/model_runner.py`
- `babyvllm/engine/outputs.py`

BabyVllm has two user-facing execution modes. Offline generation calls `LLMEngine.generate()` and blocks until all prompts finish. Online serving wraps the same synchronous engine with `AsyncLLMEngine`, `RequestTracker`, and per-request `AsyncStream` objects so each HTTP client can receive streaming chunks independently.

```mermaid
flowchart TD
  CLI["babyvllm-server CLI"] --> BuildEngine["Create AsyncLLMEngine"]
  BuildEngine --> Inject["Inject engine into api_server._engine"]
  Inject --> FastAPI["FastAPI app"]
  FastAPI --> Completion["/v1/completions"]
  FastAPI --> Chat["/v1/chat/completions"]
  Completion --> AsyncGenerate["AsyncLLMEngine.generate()"]
  Chat --> AsyncGenerate

  OfflineCaller["Offline Python caller"] --> LLMGenerate["LLMEngine.generate()"]

  AsyncGenerate --> Tracker["RequestTracker and AsyncStream"]
  Tracker --> EngineLoop["_run_engine_loop()"]
  EngineLoop --> SyncEngine["LLMEngine schedule/run_scheduled"]
  LLMGenerate --> SyncEngine

  SyncEngine --> Scheduler["Scheduler"]
  Scheduler --> Batch["ScheduledBatch: decode_sequences + prefill_sequences"]
  Batch --> ModelRunner["ModelRunner.run()"]
  ModelRunner --> Qwen3["Qwen3ForCausalLM"]
  Qwen3 --> Tokens["Sampled token ids"]
  Tokens --> Postprocess["Scheduler.postprocess()"]
  Postprocess --> OfflineOutput["Offline dict outputs + metrics"]
  Postprocess --> RequestOutput["RequestOutput chunks"]
  RequestOutput --> Tracker
  Tracker --> SSE["JSON response or SSE stream"]
```

The scheduler returns a logical batch split into Decode and Prefill sub-batches. The engine executes Decode first, then Prefill. This keeps pure Decode forwards eligible for CUDA Graph replay while allowing new Prefill work to share the same logical scheduling step.

