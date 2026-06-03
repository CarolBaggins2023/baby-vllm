"""
AsyncLLMEngine —— Online Async Inference Engine

Core Responsibilities:
  Wrap LLMEngine (synchronous, batch inference) as an asynchronous, streaming-capable online inference engine.
  Each request gets an independent AsyncStream for consuming output tokens without blocking each other.

=============================================================
Architecture Overview (Data Flow + Control Flow)
=============================================================


                        ┌─────────────────────────────────────────┐
                        │           AsyncLLMEngine                │
                        │                                         │
  API Layer             │  ┌───────────┐     ┌──────────────┐     │
  (FastAPI)             │  │ generate()│     │add_request() │     │
     │                  │  │ (async    │     │(sync method) │     │
     │  POST /generate  │  │ generator)│     │              │     │
     │  ───────────────>│  └────┬──────┘     └──────┬───────┘     │
     │                  │       │                   │             │
     │  SSE stream      │       │ 1. start loop     │ 2. register │
     │  <───────────────│       │    (lazy)         │    tracker  │
     │                  │       │                   │             │
     │                  │       │              ┌────▼──────────┐  │
     │                  │       │              │RequestTracker │  │
     │                  │       │              │               │  │
     │                  │       │              │ _new_requests │  │
     │                  │       │              │ (asyncio.Q)   │  │
     │                  │       │              │               │  │
     │                  │       │              │ _request_     │  │
     │                  │       │              │  streams:{}   │  │
     │                  │       │              │               │  │
     │                  │       │              │ new_requests_ │  │
     │                  │       │              │  event(Event) │  │
     │                  │       │              └───────┬───────┘  │
     │                  │       │                      │          │
     │                  │       │       ┌──────────────▼────────┐ │
     │                  │       │       │  _run_engine_loop()   │ │
     │                  │       │       │  (background Task)    │ │
     │                  │       │       │                       │ │
     │                  │       │       │  Loop:                │ │
     │                  │       │       │   1. drain new_reqs   │ │
     │                  │       │       │   2. create Sequence  │ │
     │                  │       │       │   3. add to scheduler │ │
     │                  │       │       │   4. check has_work   │ │
     │                  │       │       │   5. if idle: wait()  │ │
     │                  │  yield│       │   6. run_in_executor  │ │
     │                  │ <─────│       │      (engine.step)    │ │
     │                  │       │       │   7. route outputs    │ │
     │                  │       │       │   8. asyncio.sleep(0) │ │
     │                  │       │       └──────────┬────────────┘ │
     │                  │       │                  │              │
     │                  │       │       ┌──────────▼───────────┐  │
     │                  │       │       │     LLMEngine        │  │
     │                  │       │       │  (sync, CPU/GPU)     │  │
     │                  │       │       │                      │  │
     │                  │       │       │  - scheduler         │  │
     │                  │       │       │  - model_runner      │  │
     │                  │       │       │  - tokenizer         │  │
     │                  │       │       └──────────────────────┘  │
     │                  │       │                                 │
     └──────────────────┴───────┴─────────────────────────────────┘

  Data Flow Direction:
    POST request → generate() → add_request() → tracker._new_requests
                                                │
                                      _engine_step() dequeues
                                                │
                                    Create Sequence → scheduler.add_sequence()
                                                │
                                    engine.step() (thread pool)
                                                │
                                 outputs → RequestOutput → tracker routing
                                                │
                               AsyncStream.put() → generator() → SSE response

"""

from __future__ import annotations

import asyncio
import itertools
import time
from typing import AsyncGenerator, Optional, Union

# LLMEngine imported lazily inside __init__ to avoid triggering GPU/model
# dependencies during unit-test imports of AsyncLLMEngine.
from babyvllm.engine.request_tracker import RequestTracker
from babyvllm.engine.outputs import RequestOutput
from babyvllm.engine.sequence import Sequence
from babyvllm.sampling_params import SamplingParams


class AsyncLLMEngine:
    """
    baby-vllm Online Async Inference Engine.

    Wraps synchronous LLMEngine, implementing via background asyncio loop:
      - Concurrent request handling (multiple generate() coroutines run in parallel, non-blocking)
      - Streaming output (each request has independent AsyncStream with internal asyncio.Queue)
      - Lazy startup (background loop created only on first request, zero CPU when idle)

    Fundamental difference from LLMEngine.generate() (offline batch mode):
      - Offline: All prompts submitted at once → block until all complete → return list
      - Online: Requests arrive independently → independent streaming output → non-blocking between requests

    ----- Collaboration Interface with API Server ------

    Usage Example:
        engine = AsyncLLMEngine(model="/path/to/model", max_num_sequences=32)
        sampling_params = SamplingParams(max_tokens=64)

        # Single request
        async for output in engine.generate("Hello, please introduce yourself", sampling_params):
            print(f"Request {output.request_id}: {output.text}")

        # Concurrent requests
        async def main():
            tasks = [
                engine.generate("Question 1", sampling_params),
                engine.generate("Question 2", sampling_params),
            ]
            for coro in asyncio.as_completed(tasks):
                async for output in await coro:
                    print(f"[{output.request_id}] {output.text[:50]}...")

        # Shutdown
        await engine.stop()
    """

    def __init__(self, model: Optional[str] = None, engine: Optional[LLMEngine] = None, **kwargs):
        """
        Create AsyncLLMEngine instance. Does not start background loop (lazy startup, see generate()).

        Initialization Order:
          1. Create or reuse underlying synchronous LLMEngine → load model, start worker processes
          2. Create RequestTracker → prepare _new_requests queue
          3. Initialize async primitives → stop signal, mappings, counters

        Args:
            model: Model path (local directory), passed to Config(model, **kwargs).
                   Required if `engine` is not provided.
            engine: Optional pre-existing LLMEngine instance. When provided, `model` and
                    **kwargs are ignored — the passed engine is used directly.
                    Useful for sharing a single engine across sync and async test fixtures.
            **kwargs: Configuration parameters, passed to LLMEngine.__init__() → Config constructor.
                      Ignored when `engine` is provided.

        Resource Notes:
          - When creating a new engine: LLMEngine.__init__() internally starts model_runner
            processes (cleaned up via atexit)
          - When reusing an existing engine: resource lifecycle is managed by the original owner
          - Background loop Task not created (lazy startup)
          - After this method returns, engine is in "ready but not running" state
        """

        # (1) Underlying Synchronous Engine
        # Two modes:
        #   a) Pass an existing LLMEngine → reuse it (avoids double-loading model)
        #   b) Pass model path → create a new LLMEngine
        # LLMEngine is responsible for:
        #   - Model loading + multi-process Worker (ModelRunner via multiprocessing.spawn)
        #   - Tokenizer (HuggingFace AutoTokenizer: encode / decode)
        #   - Scheduler (sequence scheduling + KV Cache Block allocation)
        if engine is not None:
            self.engine = engine
        elif model is not None:
            from babyvllm.engine.llm_engine import LLMEngine
            self.engine = LLMEngine(model, **kwargs)
        else:
            raise ValueError(
                "Either `model` or `engine` must be provided to AsyncLLMEngine."
            )

        # (2) Convenient Tokenizer Reference
        # engine.tokenizer returns HuggingFace AutoTokenizer instance.
        # Provides convenient reference to avoid repeated self.engine.tokenizer calls.
        self.tokenizer = self.engine.tokenizer

        # (3) Request Tracker
        # RequestTracker is the request "router".
        self._request_tracker = RequestTracker()

        # (4) ID Generation and Mapping Tables
        # request_id is generated by AsyncLLMEngine's own counter, decoupled from Sequence.seq_id.
        #
        # Why not use Sequence.counter?
        #   1. Sequence.seq_id is auto-generated in Sequence.__init__(), cannot be known in advance
        #   2. add_request() needs request_id first to register with tracker
        #   3. Future: one request may correspond to multiple sequences (e.g., beam search, parallel sampling)
        #
        # Mapping tables are only operated in engine loop (_engine_step), following single-thread principle.
        self._request_counter = itertools.count()

        # One-way mapping: seq_id → request_id.
        # Written in _add_request_to_engine() (after Sequence creation),
        # Read and popped during output routing phase in _engine_step().
        # Why one-way only?
        #   All reverse lookup scenarios occur during output routing where seq_id is known.
        self._seq_to_request: dict[int, int] = {}

        # Reverse mapping: request_id → seq_id.
        # Written in _add_request_to_engine() when Sequence is created,
        # Read and popped in abort() and _engine_step() when processing aborted requests.
        # Why needed? abort() is called with request_id (from API layer),
        # but scheduler.abort_sequence() requires seq_id.
        self._request_to_seq: dict[int, int] = {}

        # request_id → prompt_token_ids mapping.
        # Written in add_request() (called by API layer),
        # Read and popped during output routing phase in _engine_step().
        # Why not get from Sequence when constructing RequestOutput?
        #   Sequence lifecycle is managed by scheduler after postprocess and may be cleaned up.
        #   Independent storage avoids dependency on Sequence object lifecycle.
        self._prompt_map: dict[int, list[int]] = {}

        # request_id → timing info for per-request metrics collection.
        # Written in add_request() (arrival_time), read and popped during
        # output routing phase in _engine_step() when the request finishes.
        # Structure: {"arrival_time": float}
        self._request_timings: dict[int, dict] = {}

        # (5) Background Loop and Lifecycle
        # Background loop Task reference. Initially None, lazily created on first generate() call.
        # Wraps _run_engine_loop() coroutine using asyncio.ensure_future().
        self._engine_task: Optional[asyncio.Task] = None

        # Stop signal. After set(), background loop exits on next iteration.
        # Uses asyncio.Event instead of threading.Event (entire control flow is in asyncio).
        self._stop_event = asyncio.Event()

        # (6) Current Request ID for Cancellation
        # Set in generate() after add_request() returns, before first async yield.
        # Used by API layer's CancelledError handler as a fallback when the client
        # disconnects before the first RequestOutput is yielded (engine_request_id
        # would otherwise still be None).
        # This is set BEFORE any async operation in generate(), so within a single
        # coroutine it is stable — Python asyncio is cooperative, not preemptive.
        self._current_request_id: Optional[int] = None

    @property
    def engine_started(self) -> bool:
        """
        Whether the background engine loop has started.

        Returns:
            True if _engine_task exists and is not done (i.e., background loop is running).
            Used externally to check engine status, avoid duplicate startup,
            or confirm engine started before shutdown.

        Usage Example:
            engine = AsyncLLMEngine("model_path")
            assert not engine.engine_started  # Not started after __init__
            # ... after first generate() call ...
            assert engine.engine_started      # Background loop started
        """
        return self._engine_task is not None and not self._engine_task.done()

    # =========================================================================
    # Phase 1: Request Registration (API Layer -> RequestTracker)
    # =========================================================================

    def add_request(
        self,
        sampling_params: SamplingParams,
        prompt: Optional[str] = None,
        prompt_token_ids: Optional[list[int]] = None,
    ) -> tuple:
        """
        Register a new inference request (synchronous method, called by API layer / generate()).

        This is the first phase of "two-phase request registration" - only registration, no Sequence creation:
          1. Tokenize (if string prompt provided)
          2. Generate unique request_id (using own _request_counter)
          3. Store prompt_token_ids in _prompt_map
          4. Call tracker.add_request() to enqueue request into _new_requests
          5. Return (AsyncStream, request_id)

        Second phase in _engine_step():
            drain queue from tracker -> create Sequence -> add to scheduler.

        Why delay all scheduler operations to second phase?
          LLMEngine.scheduler is not thread-safe.
          Centralizing all scheduler operations in engine loop can avoid coroutine race conditions.

        Thread Safety:
          - This method is called by API layer (asyncio event loop coroutine)
          - RequestTracker._new_requests is asyncio.Queue, coroutine-safe
          - new_requests_event.set() is coroutine-safe
          - _prompt_map writes happen in event loop, same as engine loop, no concurrent writes

        Args:
            sampling_params: Sampling parameters (temperature, max_tokens, ignore_eos, max_model_length).
            prompt: String input, mutually exclusive with prompt_token_ids.
            prompt_token_ids: Token ID list input, mutually exclusive with prompt.

        Returns:
            (stream, request_id): tuple[AsyncStream, int]
              - stream: AsyncStream object for asynchronous output consumption
              - request_id: Assigned unique request ID for caller tracking and logging

        Raises:
            ValueError: Neither prompt nor prompt_token_ids provided.
            KeyError: Duplicate request_id (guaranteed unique by itertools.count, theoretically impossible).

        Usage Example:
            engine = AsyncLLMEngine("model_path")
            sp = SamplingParams(max_tokens=64)

            # String prompt
            stream, rid = engine.add_request(
                sampling_params=sp,
                prompt="Explain quantum computing",
            )
            print(f"Request {rid} registered")
            async for output in stream.generator():
                print(output.text)

            # token ID prompt
            token_ids = engine.tokenizer.encode("Explain quantum computing")
            stream, rid = engine.add_request(
                sampling_params=sp,
                prompt_token_ids=token_ids,
            )
            async for output in stream.generator():
                print(output.text)
        """

        # (1) Tokenize: Convert string to token_ids if provided.
        #     Uses self.tokenizer (HuggingFace AutoTokenizer) for encoding.
        #     Why tokenize in add_request instead of _engine_step?
        #       - tokenizer.encode is fast CPU operation, won't block GPU
        #       - Early tokenization exposes input format errors sooner
        #       - Tokenization result needs to be stored in _prompt_map, natural place to do it
        if prompt_token_ids is None:
            if prompt is None:
                raise ValueError(
                    "Either prompt or prompt_token_ids must be provided."
                )
            prompt_token_ids = self.tokenizer.encode(prompt)

        # (2) Generate unique request_id.
        #     Uses self._request_counter (itertools.count) for auto-increment.
        #     This is AsyncLLMEngine-level ID, completely independent of Sequence.seq_id.
        request_id = next(self._request_counter)

        # (3) Store prompt_token_ids in _prompt_map.
        #     Later in output routing phase of _engine_step(),
        #     retrieve prompt_token_ids via request_id to construct RequestOutput.
        #     Entry is popped after request completes.
        self._prompt_map[request_id] = prompt_token_ids

        # (3b) Record arrival time for per-request timing metrics.
        #     Used in _engine_step() to compute TTFT/TPOT/total_time
        #     when the request's output is routed.
        self._request_timings[request_id] = {"arrival_time": time.time()}

        # (4) Register request in RequestTracker and get AsyncStream.
        #     Note: Sequence not created yet, nor moved from _new_requests to _request_streams.
        #           These operations completed when get_new_requests() is called in _engine_step().
        stream = self._request_tracker.add_request(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
        )

        # (5) Return (stream, request_id).
        #     stream for generate() async for consumption,
        #     request_id for caller to track request (logging, cancellation, etc.).
        return stream, request_id

    # =========================================================================
    # Phase 2: Request Landing (RequestTracker → Scheduler)
    # =========================================================================

    def _add_request_to_engine(self, request_data: dict) -> None:
        """
        Transfer a new request from tracker to engine scheduler.

        This is the second phase of "two-phase request registration" - all scheduler changes are centralized here:
          1. Extract prompt_token_ids and sampling_params from request_data
          2. Create Sequence object (internally gets unique seq_id from Sequence.counter)
          3. Add Sequence to Scheduler's waiting queue
          4. Build seq_id → request_id mapping (_seq_to_request)
            Note: _prompt_map already built in add_request() phase, no need to rebuild here

        Role of RequestTracker._new_requests:
          asyncio.Queue acts as bridge between API layer (asyncio coroutines) and engine loop:
            - Queue is asyncio primitive, naturally safe in event loop
            - Engine loop drains at once (while not empty: get_nowait())
            - Avoids state inconsistency from incremental processing

        Args:
            request_data: Dictionary obtained from RequestTracker.get_new_requests().
                          Structure: {"request_id": int,
                              "prompt_token_ids": list[int],
                              "sampling_params": SamplingParams}

        Usage Example (internal call):
            new_requests = self._request_tracker.get_new_requests()
            for request_data in new_requests:
                self._add_request_to_engine(request_data)
        """

        # Destructure request data from tracker
        request_id = request_data["request_id"]
        prompt_token_ids = request_data["prompt_token_ids"]
        sampling_params = request_data["sampling_params"]

        # Create Sequence object.
        seq = Sequence(
            token_ids=prompt_token_ids,
            sampling_params=sampling_params,
        )

        # Add Sequence to scheduler's waiting queue.
        # Scheduler.add_sequence() simply appends to self.waiting deque.
        # Sequence will be scheduled in next scheduler.schedule() call.
        self.engine.scheduler.add_sequence(seq)

        # Build seq_id -> request_id mapping.
        # When engine.step() returns outputs, output uses seq_id as key (since scheduler manages by seq),
        # we need this mapping to find original request_id for
        # constructing RequestOutput and routing to correct AsyncStream.
        self._seq_to_request[seq.seq_id] = request_id

        # Build reverse mapping for abort() lookups.
        # abort() is called with request_id (from API layer / CancelledError handler),
        # but scheduler.abort_sequence() requires seq_id.
        # This reverse mapping enables O(1) lookup instead of scanning _seq_to_request.
        self._request_to_seq[request_id] = seq.seq_id

    # =========================================================================
    # Phase 3: Single Engine Iteration
    # =========================================================================

    def _route_engine_outputs(
        self,
        outputs: list[tuple[int, list[int], bool]],
    ) -> None:
        """Convert engine token deltas to RequestOutput and route them."""

        request_outputs: list[RequestOutput] = []

        for seq_id, completion_token_ids, finished in outputs:
            # Look up request_id from forward mapping. Keep mappings alive until
            # the final chunk so intermediate streaming deltas can keep routing.
            if finished:
                request_id = self._seq_to_request.pop(seq_id, None)
            else:
                request_id = self._seq_to_request.get(seq_id)

            if request_id is not None:
                # Normal path: found corresponding request_id.
                if finished:
                    # Request completed: clean prompt and reverse mapping.
                    prompt_token_ids = self._prompt_map.pop(request_id, [])
                    self._request_to_seq.pop(request_id, None)
                else:
                    prompt_token_ids = self._prompt_map.get(request_id, [])
            else:
                # The sequence may have been aborted after it was scheduled.
                prompt_token_ids = []
                request_id = -1

            # Decode this step's completion delta to text.
            text = self.tokenizer.decode(completion_token_ids)

            # Compute per-request timing metrics.
            ttft = None
            tpot = None
            total_time = None
            if request_id in self._request_timings:
                timing = self._request_timings[request_id]
                now = time.time()
                timing["num_tokens"] = timing.get("num_tokens", 0) + len(completion_token_ids)
                if completion_token_ids and "first_token_time" not in timing:
                    timing["first_token_time"] = now

                if finished:
                    timing = self._request_timings.pop(request_id)
                    arrival = timing["arrival_time"]
                    completion = now
                    total_time = completion-arrival
                    first_token_time = timing.get("first_token_time")
                    num_tokens = timing.get("num_tokens", 0)
                    if first_token_time is not None:
                        ttft = first_token_time-arrival
                        tpot = (
                            (completion-first_token_time)/max(num_tokens-1, 1)
                            if num_tokens > 1 else 0.0
                        )
                    else:
                        ttft = total_time
                        tpot = 0.0

            request_outputs.append(RequestOutput(
                request_id=request_id,
                text=text,
                token_ids=completion_token_ids,
                finished=finished,
                prompt_token_ids=prompt_token_ids,
                ttft=ttft,
                tpot=tpot,
                total_time=total_time,
            ))

        self._request_tracker.process_step_outputs(request_outputs)

    async def _engine_step(self) -> bool:
        """
        Execute one complete inference step of the engine.

        Full cycle: abort processing → new request admission → scheduling →
        GPU execution → result routing. Updated for Phase 5 with error handling
        and abort support.

        Execution Order:
          1. Drain aborted requests + new requests from RequestTracker
          2. Process aborts first (free KV cache blocks before scheduling)
          3. Land new requests to scheduler (with per-request error handling)
          4. Check if there is work to do
          5. Schedule one logical batch
          6. Execute Decode sub-batch first and route its outputs immediately
          7. Execute Prefill sub-batch and route any first-token outputs

        Error Handling Strategy (Two-Level):
          Level 1 - Per-request errors: If adding a single request fails,
                    propagate exception to that request's stream only.
                    Other requests continue unaffected.
          Level 2 - Catastrophic errors: If scheduling or model execution fails,
                    all active requests get the exception.
                    All scheduler state is cleaned up.

        Returns:
            True if a scheduler iteration was executed, False if idle.

        Decode output is routed before Prefill starts. This preserves streaming
        TTFT even when the same logical scheduler iteration also admits Prefill.
        """

        # =====================================================================
        # (1) Drain aborted requests AND new requests from tracker
        # =====================================================================
        # get_new_and_aborted_requests() processes both queues atomically:
        #   - Aborted IDs are deduplicated into a set
        #   - New requests that happen to also be in the abort set are rejected
        #     immediately (extreme race: add + immediate cancel)
        #   - Surviving new requests are registered in _request_streams
        new_requests, aborted_request_ids = \
            self._request_tracker.get_new_and_aborted_requests()

        # =====================================================================
        # (2) Process aborted requests FIRST
        # =====================================================================
        # Why before new request landing?
        #   Aborting frees KV cache blocks. Processing aborts first maximizes
        #   available blocks for new requests in the same iteration.
        #   This improves scheduling efficiency under churn.
        for request_id in aborted_request_ids:
            # Look up seq_id from reverse mapping
            seq_id = self._request_to_seq.pop(request_id, None)
            if seq_id is not None:
                # Abort in scheduler (free blocks, remove from deques)
                self.engine.scheduler.abort_sequence(seq_id)
                # Clean forward mapping
                self._seq_to_request.pop(seq_id, None)
            # Clean prompt map and timing data
            self._prompt_map.pop(request_id, None)
            self._request_timings.pop(request_id, None)

        # =====================================================================
        # (3) Land new requests to scheduler
        # =====================================================================
        for request_data in new_requests:
            try:
                self._add_request_to_engine(request_data)
            except Exception as e:
                # Level 1: Per-request error.
                # If landing one request fails (e.g., invalid data), propagate
                # the exception to just that request's stream. Other requests
                # continue normally.
                self._request_tracker.process_exception(
                    request_data["request_id"], e
                )

        # =====================================================================
        # (4) Check if there is work to do
        # =====================================================================
        # "Has work" = new requests landed this round OR scheduler has active sequences.
        #
        # Note on ordering: We check AFTER processing aborts because abort may
        # have emptied the scheduler. But if new_requests landed successfully,
        # they're now in the scheduler's waiting queue, so is_finished() returns False.
        has_work = bool(new_requests) or not self.engine.scheduler.is_finished()

        if not has_work:
            return False

        # =====================================================================
        # (5) Execute one logical step with Decode routed before Prefill
        # =====================================================================
        loop = asyncio.get_running_loop()
        try:
            batch = await loop.run_in_executor(
                None,
                self.engine.schedule,
            )

            if batch.decode_sequences:
                outputs = await loop.run_in_executor(
                    None,
                    self.engine.run_scheduled,
                    batch.decode_sequences,
                )
                self._route_engine_outputs(outputs)

                # Give streaming clients a chance to consume Decode tokens while
                # the following Prefill sub-batch runs in the executor.
                await asyncio.sleep(0)

            if batch.prefill_sequences:
                outputs = await loop.run_in_executor(
                    None,
                    self.engine.run_scheduled,
                    batch.prefill_sequences,
                )
                self._route_engine_outputs(outputs)
        except Exception as e:
            # Level 2: Catastrophic engine failure.
            # This means the model runner itself failed (GPU OOM, CUDA error,
            # multiprocessing failure, etc.). All in-flight requests are affected
            # because the engine operates on all sequences as a batch.

            # Propagate exception to ALL active request streams
            self._request_tracker.process_exception_all(e)

            # Abort all sequences still in scheduler (frees KV cache blocks)
            # Must iterate over snapshots since abort_sequence mutates the deques.
            # Example: if running=[seq1, seq2, seq3] and we abort seq1,
            #          the deque shifts; iterating a snapshot avoids index errors.
            for seq in list(self.engine.scheduler.running) + \
                       list(self.engine.scheduler.waiting):
                self.engine.scheduler.abort_sequence(seq.seq_id)

            # Clean up ALL internal mappings (nothing left to track)
            self._seq_to_request.clear()
            self._request_to_seq.clear()
            self._prompt_map.clear()
            self._request_timings.clear()

            # Re-raise to crash the background loop.
            # The API layer will see the exception through the streams
            # (already propagated above). The loop exits, and a new one
            # will be lazily created on the next generate() call.
            raise

        return True

    # =========================================================================
    # Phase 4: Background Engine Main Loop
    # =========================================================================

    async def _run_engine_loop(self):
        """
        Background engine main loop — runs continuously as independent asyncio Task until stop signal received.

        Loop logic:
          while not _stop_event:
              1. _engine_step()           # One complete schedule-execute-route cycle
              2. If has work → asyncio.sleep(0)  # Yield control to event loop
              3. If no work → await wait_for_new_requests()  # Block waiting for new requests

        Idle Waiting Design:
          Uses asyncio.Event instead of busy-wait.
          wait_for_new_requests() internally awaits tracker.new_requests_event.wait().
          When new request arrives (add_request → set() event), event loop immediately wakes this task.
          Ensures low latency with zero CPU idle.

        Wake-up Flow:
          API layer add_request()
            -> tracker._new_requests.put_nowait((stream, req_data))
            -> tracker.new_requests_event.set()      # wake up engine loop
                                                     │
          Engine loop wait_for_new_requests()        │
            -> await new_requests_event.wait() ──────┘ (woken up)
            -> new_requests_event.clear()
            -> Return, continue loop

        Exit Condition:
          After _stop_event is set(), loop exits at top of next iteration.
          Externally set via stop() method which sets _stop_event and cancels this Task.

        Why only sleep(0) when working?
          asyncio.sleep(0) suspends current coroutine to end of event loop queue,
          giving other ready coroutines (e.g., stream.generator() consumers) a chance to run.
          This is important for fairness — without yielding, consecutive steps could monopolize the event loop,
          causing client output delays.
          But sleep(0) doesn't actually wait — engine enters next step ASAP, minimizing throughput loss.

        Loop Flow Diagram:
            ┌──────────────────────────────────────────────┐
            │        _run_engine_loop()                    │
            │                                              │
            │  ┌──────────────────────────────────────┐    │
            │  │ while not _stop_event:               │    │
            │  │                                      │    │
            │  │   had_work = _engine_step()          │    │
            │  │         │                            │    │
            │  │    ┌────┴────┐                       │    │
            │  │    │         │                       │    │
            │  │  True      False                     │    │
            │  │    │         │                       │    │
            │  │  sleep(0)  wait_for_new_reqs()       │    │
            │  │  (yield)   (block until new request) │    │
            │  │    │         │                       │    │
            │  │    └────┬────┘                       │    │
            │  │         │                            │    │
            │  │         └---> next iteration         │    │
            │  └──────────────────────────────────────┘    │
            └──────────────────────────────────────────────┘
        """

        while not self._stop_event.is_set():
            # (1) Execute one engine step.
            #     had_work=True  -> engine.step() was executed
            #     had_work=False -> completely idle (no sequences in scheduler + no new requests)
            had_work = await self._engine_step()

            if not had_work:
                # (2.1) Idle: Block waiting for new requests.
                #      wait_for_new_requests() internally:
                #        - If _new_requests queue is already non-empty (add_request called during step),
                #          method checks has_new_requests() -> returns immediately
                #        - Otherwise await new_requests_event.wait() -> block
                #      event is set() in add_request() -> wakes immediately
                await self._request_tracker.wait_for_new_requests()
            else:
                # (2.2) Work completed: Briefly yield control.
                #      asyncio.sleep(0) suspends current coroutine to end of event loop queue,
                #      allowing other ready coroutines to execute before continuing to next step.
                await asyncio.sleep(0)

    # =========================================================================
    # Phase 5: External API — Streaming Generation
    # =========================================================================

    async def generate(
        self,
        prompt: Union[str, list[int]],
        sampling_params: SamplingParams,
        request_id: Optional[int] = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        """
        Async generator — main entry point for API layer.

        Each generate() call corresponds to an independent inference request.
        Caller consumes RequestOutput one by one via async for (in current baby-vllm implementation,
        each request typically yields once since engine.step() only outputs when sequence completes).

        Internal Flow:
          1. Lazily start background engine loop (if not started yet)
          2. Call add_request() to register request, get (stream, request_id)
          3. Iterate output from stream.generator() via async for
          4. Yield one RequestOutput to caller each time

        Concurrent Call Scenario:
          Multiple generate() coroutines can run concurrently:
            - First generate() creates background loop
            - Subsequent generate() reuse existing loop
            - All requests queued in same scheduler
            - Output routed independently via respective AsyncStreams

        Args:
            prompt: Input prompt.
                    - str: Text format, automatically encoded by tokenizer
                    - list[int]: Token ID list format, used directly
            sampling_params: Sampling parameters.
            request_id: Optional externally specified request ID.
                        If None, assigned internally by add_request().
                        Externally specified mainly for request tracking scenarios
                        (e.g., correlation with upstream systems).


        Yields:
            RequestOutput: Output object containing request_id, text, token_ids, finished=True,
                           prompt_token_ids.

        Raises:
            ValueError: prompt type is not str or list[int].
            KeyError: Duplicate request_id (if externally specifying an existing ID).
            asyncio.CancelledError: Caller cancels generation (e.g., client disconnects),
                                    handled via AsyncStream's aclose mechanism.

        Usage Example:
            engine = AsyncLLMEngine("model_path")
            sp = SamplingParams(max_tokens=64)

            # String input
            async for output in engine.generate("Explain quantum computing", sp):
                print(f"[Request {output.request_id}] {output.text}")

            # Token ID input
            token_ids = engine.tokenizer.encode("Explain quantum computing")
            async for output in engine.generate(token_ids, sp):
                print(f"[Request {output.request_id}] {output.text}")

            # Concurrent
            async def concurrent_demo():
                async def worker(prompt, idx):
                    async for out in engine.generate(prompt, sp):
                        print(f"Worker {idx}: {out.text[:50]}...")

                await asyncio.gather(
                    worker("Question 1", 1),
                    worker("Question 2", 2),
                    worker("Question 3", 3),
                )
        """

        # (1) Lazy startup of background loop
        # Create background Task only on first generate() call.
        # Create new Task if previous one is done (e.g., after stop()),
        # OR if the previous Task was created on a different event loop.
        #
        # Why not start in __init__?
        #   - Engine may only be used for offline inference (via LLMEngine.generate() sync interface),
        #     in which case background loop is not needed
        #   - Clear lifecycle: engine starts = first request arrives
        #
        # Why check done() instead of just is None?
        #   - After stop(), _engine_task still holds reference (not None), but is done
        #   - User calling generate() again should auto-restart without explicit reinitialization
        #
        # Why use asyncio.ensure_future()?
        #   - ensure_future() wraps coroutine as Task and schedules it immediately
        #   - Equivalent to create_task() (Python 3.7+), but ensure_future is more general
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        needs_recreate = (
            self._engine_task is None
            or self._engine_task.done()
            or (
                current_loop is not None
                and self._engine_task.get_loop() is not current_loop
            )
        )
        if needs_recreate:
            self._stop_event = asyncio.Event()
            # Recreate RequestTracker to bind its asyncio primitives
            # (new_requests_event, queues) to the current event loop.
            # Without this, primitives created in __init__ (possibly a
            # different loop) cause RuntimeError when Event.wait() checks
            # loop affinity via _get_loop().
            self._request_tracker = RequestTracker()
            self._engine_task = asyncio.ensure_future(
                self._run_engine_loop()
            )

        # (2) Process prompt format and register request
        # add_request() is synchronous (only memory operations: encoding + enqueue),
        # does not trigger actual model inference. Inference runs asynchronously in background loop.
        if isinstance(prompt, str):
            stream, rid = self.add_request(
                sampling_params=sampling_params,
                prompt=prompt,
            )
        elif isinstance(prompt, list):
            stream, rid = self.add_request(
                sampling_params=sampling_params,
                prompt_token_ids=prompt,
            )
        else:
            raise ValueError(
                f"prompt must be str or list[int], got {type(prompt)}."
            )

        # (3) Expose request_id for cancellation fallback BEFORE first async yield.
        # If a client disconnects during prefill (before the first RequestOutput
        # is yielded), the API layer's CancelledError handler needs engine_request_id
        # to call abort() and free KV cache blocks. By storing it here synchronously
        # (before any await), we guarantee _current_request_id is available when
        # CancelledError arrives, even during the very first __anext__() call.
        self._current_request_id = rid

        # (4) Async consume stream and yield one by one
        # stream.generator() returns _AsyncStreamIter object.
        #
        # Note: In current implementation, engine.step() only produces output when sequence completes,
        # so each request typically has only 1 RequestOutput.
        # Using async for loop (instead of awaiting single value) reserves compatibility
        # for future incremental output support (yield per token).
        async for output in stream.generator():
            yield output

    # =========================================================================
    # Phase 6: Request Cancellation
    # =========================================================================

    def abort(self, request_id: int) -> None:
        """
        Abort a request by its internal engine request_id.

        Performs full cleanup across all layers:
          1. RequestTracker: finish the stream with CancelledError
          2. Scheduler: abort the underlying sequence, free KV cache blocks
          3. Internal mappings: clean up _seq_to_request, _request_to_seq, _prompt_map

        Idempotent: safe to call multiple times for the same request_id.
        All dict pop() operations return None on missing keys, which are silently ignored.

        Why this exists:
          When a client disconnects during streaming (CancelledError in API layer),
          the engine must release GPU resources (KV cache blocks) and clean up
          internal state. Without explicit cleanup, aborted sequences continue to
          consume KV cache blocks indefinitely, starving new requests.

        Called by:
          - api_server.py when client disconnects (CancelledError handler)
          - _engine_step() when processing aborted requests from tracker

        Args:
            request_id: Internal engine request ID (integer, generated by
                        _request_counter in add_request()).

        Example:
            # Client disconnects during streaming
            engine.abort(request_id=5)
            # → tracker finishes stream with CancelledError
            # → scheduler frees KV cache blocks
            # → internal mappings cleaned up
        """
        # (1) Tracker cleanup: finishes stream, puts request_id in _aborted_requests.
        #     The stream consumer (async for loop in API layer) will receive
        #     a CancelledError when it reads from the queue.
        #     abort_request() is idempotent: calling it on an already-finished
        #     stream simply adds to _aborted_requests (which will be deduplicated
        #     by get_new_and_aborted_requests).
        self._request_tracker.abort_request(
            request_id,
            exception=asyncio.CancelledError(),
        )

        # (2) Scheduler cleanup: free KV cache blocks.
        #     Look up seq_id from reverse mapping (request_id → seq_id).
        #     pop() returns None if request_id not in mapping (already cleaned up
        #     in a previous abort call or the request was never fully landed).
        seq_id = self._request_to_seq.pop(request_id, None)
        if seq_id is not None:
            # Abort the sequence in scheduler: marks FINISHED, frees blocks,
            # removes from waiting/running deques.
            self.engine.scheduler.abort_sequence(seq_id)
            # Also clean the forward mapping (seq_id → request_id).
            # This prevents stale entries if _engine_step() later tries to
            # look up request_id from a seq_id that no longer exists.
            self._seq_to_request.pop(seq_id, None)

        # (3) Prompt map cleanup.
        #     Remove stored prompt_token_ids for this request.
        #     Pop is safe even if request was never added to _prompt_map.
        self._prompt_map.pop(request_id, None)

        # (4) Timing data cleanup.
        #     Remove timing info for this request to prevent memory leaks.
        #     Pop is safe even if request was never added to _request_timings.
        self._request_timings.pop(request_id, None)

    def get_stats(self) -> dict[str, dict[str, int]]:
        """Return engine instrumentation counters."""

        return self.engine.get_stats()

    # =========================================================================
    # Lifecycle Management
    # =========================================================================

    async def stop(self):
        """
        Stop the async engine.

        Stop Order:
          1. set() _stop_event -> notify background loop to exit while loop
          2. set() new_requests_event -> wake up loop possibly blocked in wait_for_new_requests()
                                         (setting _stop_event alone cannot interrupt asyncio.Event.wait())
          3. cancel() _engine_task -> force interrupt current await (including run_in_executor)
          4. await _engine_task -> wait for Task cleanup before exit
          5. Note: Underlying LLMEngine resource cleanup (model_runner worker processes)
             is handled automatically via atexit registration in LLMEngine.exit()

        Why need to both set() event and cancel() task?
          - _stop_event.set() only affects while condition check, cannot interrupt await blocking
          - new_requests_event.set() wakes up wait_for_new_requests() blocking
          - task.cancel() injects CancelledError at task's current await point,
            can interrupt run_in_executor and any other await

        Usage Example:
            engine = AsyncLLMEngine("model_path")
            try:
                async for output in engine.generate("Hello", sp):
                    print(output.text)
            finally:
                # Ensure background loop stops
                await engine.stop()
            # On program exit, atexit automatically calls engine.exit() to cleanup worker processes

        Notes:
          - stop() does not close underlying LLMEngine worker processes (handled by atexit)
          - After stop(), calling generate() again will auto-restart (lazy startup)
          - If there are incomplete requests, their streams are cleaned up by RequestTracker.abort_request()
        """

        # If background loop never started, nothing to stop
        if self._engine_task is None:
            return

        # (1) Send stop signal.
        #     Engine loop checks at while not self._stop_event.is_set(),
        #     after set() it exits while loop at top of next iteration.
        self._stop_event.set()

        # (2) Wake up background loop possibly blocked in wait_for_new_requests().
        #     Idle branch of _run_engine_loop():
        #       await self._request_tracker.wait_for_new_requests()
        #     Underlying implementation: await new_requests_event.wait() — blocks until set() by other coroutine
        #     Setting _stop_event alone cannot interrupt this wait (asyncio.Event doesn't support timeout cancel),
        #     so must explicitly set() new_requests_event to wake up loop.
        self._request_tracker.new_requests_event.set()

        # (c) Cancel background Task.
        #     cancel() injects CancelledError at task's next await point:
        #       - If in wait_for_new_requests() → interrupts immediately
        #       - If in run_in_executor() → doesn't interrupt thread, but raises exception at await
        #     After cancel(), task enters CANCELLED state and won't run new iterations.
        self._engine_task.cancel()

        # (d) Wait for Task to fully exit.
        #     CancelledError is expected result of cancel(), normal exit.
        try:
            await self._engine_task
        except asyncio.CancelledError:
            # Normal: cancel() causes task to raise CancelledError
            pass
