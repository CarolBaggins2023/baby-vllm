# Scheduler

## Source Modules

- `babyvllm/engine/scheduler.py`
- `babyvllm/engine/sequence.py`
- `babyvllm/engine/block_manager.py`
- `babyvllm/engine/llm_engine.py`

The scheduler owns request admission, Decode/Prefill selection, KV block allocation, preemption, and sequence state updates. It returns a `ScheduledBatch` containing separate Decode and Prefill lists. The engine runs these lists as separate physical forwards.

## Decode-First Logical Scheduling

```mermaid
flowchart TD
  Start["schedule()"] --> Decode["_schedule_decode_batch()"]
  Decode --> DecodeBudget["Decode uses one token per decode sequence"]
  DecodeBudget --> Remaining["Compute remaining sequence and token budget"]
  Remaining --> HasPrefill{"Pending Prefill work and budget left?"}
  HasPrefill -->|No| Build["Build ScheduledBatch"]
  HasPrefill -->|Yes| Reserve["Reserve one future block per active Decode sequence"]
  Reserve --> Prefill["_schedule_prefill_batch()"]
  Prefill --> Build
  Build --> Stats["_record_batch_stats(): pure_decode, pure_prefill, mixed"]
  Stats --> Return["Return decode_sequences + prefill_sequences"]
```

Decode sequences are selected before Prefill so online streaming can keep token latency low. If there is still budget, the same logical step may also admit or continue Prefill work, producing a mixed logical batch.

## Prefill Selection

```mermaid
flowchart TD
  PrefillStart["_schedule_prefill_batch()"] --> Cap["Cap max_tokens by max_prefill_tokens_per_step"]
  Cap --> ContinueRunning["Continue running chunked-prefill sequences first"]
  ContinueRunning --> RunningLoop{"Running prefill sequence fits?"}
  RunningLoop -->|Yes| AllocateRunning["allocate_chunk(), set seq.chunk_size"]
  RunningLoop -->|No| RequeueRunning["Keep sequence in running queue"]
  AllocateRunning --> RequeueRunning
  RequeueRunning --> AdmitWaiting["Admit waiting requests FCFS"]
  AdmitWaiting --> WaitingFits{"Head waiting request fits?"}
  WaitingFits -->|Yes| AllocateWaiting["allocate_chunk(), status=RUNNING, move waiting -> running"]
  AllocateWaiting --> AdmitWaiting
  WaitingFits -->|No| Stop["Stop admission without skipping later requests"]
```

`max_prefill_chunk_size` prevents one long prompt from occupying the entire Prefill window. Already-running chunked prefills are continued before admitting new requests so resident partial work can make progress.

## Decode Allocation And Preemption

```mermaid
flowchart TD
  Candidate["Decode candidate from running queue"] --> IsDecode{"num_computed_tokens == num_tokens - 1?"}
  IsDecode -->|No| KeepRunning["Append back to running"]
  IsDecode -->|Yes| NeedOne["Need chunk_size = 1"]
  NeedOne --> CanAlloc{"BlockManager.can_allocate_chunk()?"}
  CanAlloc -->|Yes| Alloc["allocate_chunk(), set chunk_size=1"]
  Alloc --> Schedule["Add to decode_sequences"]
  CanAlloc -->|No| HasOther{"Other running sequence exists?"}
  HasOther -->|Yes| PreemptOther["preempt(running.pop())"]
  HasOther -->|No| PreemptSelf["preempt(candidate)"]
  PreemptOther --> CanAlloc
  PreemptSelf --> Stop["Stop scheduling this candidate"]
```

Preemption deallocates the sequence KV blocks, resets `num_computed_tokens` to zero, marks the sequence `WAITING`, and pushes it to the front of the waiting queue. Recovery uses `num_tokens`, not only `num_prompt_tokens`, so generated tokens are replayed as part of the reconstructed context.

## Postprocess State Transitions

```mermaid
stateDiagram-v2
  [*] --> WAITING: "Sequence created"
  WAITING --> RUNNING: "Prefill chunk allocated"
  RUNNING --> RUNNING: "Chunked prefill output ignored"
  RUNNING --> RUNNING: "Prompt complete, append generated token"
  RUNNING --> FINISHED: "EOS, max_tokens, max_model_length, or zero remaining tokens"
  RUNNING --> WAITING: "Preempt and deallocate KV blocks"
  FINISHED --> [*]
```

```mermaid
flowchart TD
  Post["postprocess(seqs, token_ids)"] --> Inc["num_computed_tokens += seq.chunk_size"]
  Inc --> PromptDone{"num_computed_tokens >= num_tokens?"}
  PromptDone -->|No| Silent["Still chunked prefill: discard sampled token"]
  PromptDone -->|Yes| MaxBefore{"num_completion_tokens >= max_tokens before append?"}
  MaxBefore -->|Yes| FinishEmpty["Finish, deallocate, emit empty final delta"]
  MaxBefore -->|No| Append["append_token(token_id)"]
  Append --> StopCheck{"EOS or max_tokens or max_model_length?"}
  StopCheck -->|Yes| Finish["Finish, deallocate, remove from running"]
  StopCheck -->|No| Continue["Keep running for next Decode"]
  FinishEmpty --> Emit["Emit (seq_id, tokens, finished)"]
  Finish --> Emit
  Continue --> Emit
```

Chunked Prefill may sample tokens before the prompt is complete, but those tokens are invalid as completion output. BabyVllm discards them until the last prefill chunk has been processed.
