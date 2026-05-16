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

import asyncio
import itertools
from typing import AsyncGenerator, Optional, Union

from babyvllm.engine.llm_engine import LLMEngine
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

    def __init__(self, model: str, **kwargs):
        """
        Create AsyncLLMEngine instance. Does not start background loop (lazy startup, see generate()).

        Initialization Order:
          1. Create underlying synchronous LLMEngine → load model, start worker processes
          2. Create RequestTracker → prepare _new_requests queue
          3. Initialize async primitives → stop signal, mappings, counters

        Args:
            model: Model path (local directory), passed to Config(model, **kwargs)
            **kwargs: Configuration parameters, passed to LLMEngine.__init__() → Config constructor.

        Resource Notes:
          - LLMEngine.__init__() internally starts model_runner processes (cleaned up via atexit)
          - Background loop Task not created (lazy startup)
          - After this method returns, engine is in "ready but not running" state
        """

        # (1) Underlying Synchronous Engine
        # LLMEngine is responsible for:
        #   - Model loading + multi-process Worker (ModelRunner via multiprocessing.spawn)
        #   - Tokenizer (HuggingFace AutoTokenizer: encode / decode)
        #   - Scheduler (sequence scheduling + KV Cache Block allocation)
        self.engine = LLMEngine(model, **kwargs)

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

        # request_id → prompt_token_ids mapping.
        # Written in add_request() (called by API layer),
        # Read and popped during output routing phase in _engine_step().
        # Why not get from Sequence when constructing RequestOutput?
        #   Sequence lifecycle is managed by scheduler after postprocess and may be cleaned up.
        #   Independent storage avoids dependency on Sequence object lifecycle.
        self._prompt_map: dict[int, list[int]] = {}

        # (5) Background Loop and Lifecycle
        # Background loop Task reference. Initially None, lazily created on first generate() call.
        # Wraps _run_engine_loop() coroutine using asyncio.ensure_future().
        self._engine_task: Optional[asyncio.Task] = None

        # Stop signal. After set(), background loop exits on next iteration.
        # Uses asyncio.Event instead of threading.Event (entire control flow is in asyncio).
        self._stop_event = asyncio.Event()

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

    # =========================================================================
    # Phase 3: Single Engine Iteration
    # =========================================================================

    async def _engine_step(self) -> bool:
        """
        Execute one complete inference step of the engine.

        This is the core of the background loop - each iteration covers the complete chain:
        "new request admission -> scheduling -> GPU execution -> result routing".

        Execution Order:
          1. Drain new request queue from RequestTracker (_new_requests → scheduler)
          2. Check if there is work to do (new requests or active sequences in scheduler)
          3. Return False if no work
          4. Execute engine.step() in thread pool if there is work
          5. Process completed sequences returned by step(), construct RequestOutput and route to corresponding stream

        About engine.step() return value:
          Current LLMEngine.step() only returns completed sequences (filtered by seq.is_finished).
          This means:
            - Prefill phase: outputs is usually [] (sequence not completed yet)
            - Intermediate decode steps: outputs is [] (still generating token by token)
            - Final step: outputs contains (seq_id, completion_token_ids) (EOS/reached max_tokens)
          Therefore, each request typically has only one RequestOutput (finished=True),
          returning all generated text at once when request completes.

        Returns:
            True:  engine.step() executed this round (work available)
            False: idle this round (empty scheduler and no new requests)

        Concurrency Safety:
            This method calls engine.step() in thread pool, but engine.step() internally
            accesses no AsyncLLMEngine state (only scheduler, model_runner, tokenizer).
            Therefore, all AsyncLLMEngine state access (_seq_to_request, _prompt_map, etc.)
            is naturally single-threaded.
        """

        # (1) Drain new request queue
        # get_new_requests() drains tracker._new_requests queue:
        #   - For each (stream, request_data), register stream in _request_streams
        #   - Return request_data list
        # This is the only place engine loop reads _new_requests, ensuring FIFO order.
        new_requests = self._request_tracker.get_new_requests()

        # "Land" each new request to engine scheduler.
        for request_data in new_requests:
            self._add_request_to_engine(request_data)

        # (2) Check if there is work
        # "Has work" defined as any of:
        #   a) New requests arrived this round (already added to scheduler waiting queue above)
        #   b) Still waiting or running sequences in scheduler
        #
        # Scheduler.is_finished() returns True if and only if:
        #   Both self.waiting and self.running deques are empty
        #
        # Note: Even if scheduler.is_finished() is True,
        # if new_requests arrived this round (non-empty), they were added to waiting queue above,
        # so is_finished() returns False. Explicit bool(new_requests) check makes it clearer.
        has_work = bool(new_requests) or not self.engine.scheduler.is_finished()

        # (3) Early return if no work
        if not has_work:
            return False

        # (4) Execute engine.step() in thread pool
        #
        # Why use run_in_executor?
        #   model_runner.call() involves GPU operations and inter-process communication (shared memory + Event),
        #   which is synchronous blocking (may take tens to hundreds of ms).
        #   Executing in default ThreadPoolExecutor frees the asyncio event loop to handle other coroutines
        #   (e.g., stream.generator() consumption from other generate(), new request add_request(), etc.).
        #
        # Why not use ProcessPoolExecutor?
        #   engine.step() needs to communicate with worker processes (via shared memory + Event),
        #   these IPC mechanisms are bound to the main process. Forked child processes cannot use them properly.
        #
        # Note: Python GIL is released in PyTorch C extensions (CUDA operations), so thread pool is effective.
        loop = asyncio.get_running_loop()
        # None = use default ThreadPoolExecutor (auto-created in Python 3.8+)
        outputs, is_prefill = await loop.run_in_executor(
            None,
            self.engine.step,
        )

        # (5) Construct RequestOutput and route
        # engine.step() only returns outputs for completed (FINISHED) sequences.
        # For each completed sequence:
        #   - Reverse lookup request_id via _seq_to_request
        #   - Get prompt_token_ids from _prompt_map
        #   - Decode completion_token_ids to full text
        #   - Construct RequestOutput and add to list
        #   - Pop from mapping tables (sequence completed, mapping no longer needed)
        request_outputs: list[RequestOutput] = []

        for seq_id, completion_token_ids in outputs:
            # Lookup and remove seq_id -> request_id mapping.
            # Use pop() instead of get():
            #   - Sequence completed, mapping no longer needed
            #   - Clean up promptly to prevent memory leak
            #   - Default None handles edge case (seq_id not in mapping due to exception)
            request_id = self._seq_to_request.pop(seq_id, None)

            if request_id is not None:
                # Normal path: Found corresponding request_id.
                # Get and remove prompt_token_ids from _prompt_map.
                # Why pop instead of get?
                #   - Request completed, prompt info written to RequestOutput, no longer needed
                #   - Clean up promptly to release memory (especially important for long prompts)
                prompt_token_ids = self._prompt_map.pop(request_id, [])
            else:
                # Abnormal path: seq_id not found in _seq_to_request.
                # Possible reasons:
                #   - Stale sequence before engine initialization (theoretically impossible)
                #   - Mapping lost due to edge cases like concurrent cancellation
                # Degradation handling: Use empty prompt and placeholder request_id.
                # This RequestOutput will still be routed by tracker,
                # but request_id=-1 will almost certainly not be found in _request_streams,
                # and tracker.process_request_output() will silently drop it (safe degradation).
                prompt_token_ids = []
                request_id = -1

            # Decode: Convert completion_token_ids back to text.
            # Since engine.step() returns complete completion tokens
            # (from first generated token to end token), one decode yields full output text.
            text = self.tokenizer.decode(completion_token_ids)

            # Construct RequestOutput.
            # finished is always True because engine.step() only outputs filtered by seq.is_finished.
            # If engine.step() changes to output incremental tokens (streaming) in the future,
            # this needs to set finished based on actual completion status.
            request_outputs.append(RequestOutput(
                request_id=request_id,
                text=text,
                token_ids=completion_token_ids,
                finished=True,
                prompt_token_ids=prompt_token_ids,
            ))

        # Batch routing: Pass all RequestOutputs to tracker at once.
        # RequestTracker.process_step_outputs() internally:
        #   - Calls process_request_output() for each RequestOutput
        #   - process_request_output() puts output into corresponding stream._queue
        #   - If output.finished=True -> stream.finish() -> put STOP_ITERATION
        #     -> Also pop from _request_streams for cleanup
        self._request_tracker.process_step_outputs(request_outputs)

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
        # Create new Task if previous one is done (e.g., after stop()).
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
        if self._engine_task is None or self._engine_task.done():
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

        # (3) Async consume stream and yield one by one
        # stream.generator() returns _AsyncStreamIter object.
        #
        # Note: In current implementation, engine.step() only produces output when sequence completes,
        # so each request typically has only 1 RequestOutput.
        # Using async for loop (instead of awaiting single value) reserves compatibility
        # for future incremental output support (yield per token).
        async for output in stream.generator():
            yield output

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
