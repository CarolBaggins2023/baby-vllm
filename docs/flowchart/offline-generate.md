# Offline Generate

## Source Modules

- `babyvllm/engine/llm_engine.py`
- `babyvllm/engine/scheduler.py`
- `babyvllm/engine/model_runner.py`
- `babyvllm/engine/sequence.py`
- `babyvllm/sampling_params.py`

`LLMEngine.generate()` prepares a fixed list of prompts, enqueues every request, then repeatedly calls `step()` until the scheduler has no waiting or running sequences. It also collects throughput, memory, GPU utilization, TTFT, and TPOT metrics.

```mermaid
flowchart TD
  Start["LLMEngine.generate(prompts, sampling_params)"] --> IsDP{"data_parallel_size > 1?"}
  IsDP -->|Yes| DispatchDP["_generate_data_parallel()"]
  DispatchDP --> DPOutputs["Restore original prompt order and aggregate metrics"]

  IsDP -->|No| Normalize["_normalize_generation_inputs()"]
  Normalize --> PrepareEach["_prepare_request() for each prompt"]
  PrepareEach --> Validate["Validate sampling params, prompt ids, model length, KV capacity"]
  Validate --> Enqueue["_enqueue_request(): Sequence -> scheduler.waiting"]
  Enqueue --> Loop{"scheduler.is_finished()?"}
  Loop -->|No| Step["step()"]
  Step --> Schedule["schedule(): build ScheduledBatch"]
  Schedule --> DecodeRun["run_scheduled(decode_sequences)"]
  DecodeRun --> PrefillRun["run_scheduled(prefill_sequences)"]
  PrefillRun --> Collect["Collect token deltas and timing stats"]
  Collect --> Loop
  Loop -->|Yes| DecodeText["tokenizer.decode(generated_tokens)"]
  DecodeText --> Metrics["Build throughput, memory, GPU, TTFT, TPOT metrics"]
  Metrics --> Return["Return outputs and metrics"]
```

## Step Details

```mermaid
sequenceDiagram
  participant Caller as "Offline caller"
  participant Engine as "LLMEngine"
  participant Scheduler as "Scheduler"
  participant Runner as "ModelRunner"

  Caller->>Engine: "generate(prompts, sampling_params)"
  Engine->>Engine: "prepare and enqueue Sequence objects"
  loop "until scheduler is finished"
    Engine->>Scheduler: "schedule()"
    Scheduler-->>Engine: "ScheduledBatch"
    opt "Decode sub-batch exists"
      Engine->>Runner: "call('run', decode_sequences)"
      Runner-->>Engine: "sampled token ids"
      Engine->>Scheduler: "postprocess(decode_sequences, ids)"
    end
    opt "Prefill sub-batch exists"
      Engine->>Runner: "call('run', prefill_sequences)"
      Runner-->>Engine: "sampled token ids"
      Engine->>Scheduler: "postprocess(prefill_sequences, ids)"
    end
    Engine->>Engine: "update generated token and metrics maps"
  end
  Engine-->>Caller: "outputs, metrics"
```

Offline mode does not use `RequestOutput`. The scheduler returns `(seq_id, token_ids, finished)` tuples, and `generate()` converts final token lists into plain dictionaries.

