"""
Request Tracker - The "router" for online inference.

Core Responsibilities:
  1. Create independent AsyncStream (per-request asyncio.Queue) for each request
  2. Route RequestOutput from engine step() to the corresponding request's stream
  3. Manage request lifecycle: Create → Running → Complete/Cancel

Data Flow Overview:
  ┌─────────────┐     ┌─────────────────┐     ┌──────────────┐
  │  API Layer  │────>│  RequestTracker │────>│  LLMEngine   │
  │  (FastAPI)  │     │  (this file)    │     │  (step/sched)│
  └─────────────┘     └─────────────────┘     └──────────────┘
         │                     │                       │
         │  add_request()      │  get_new_requests()   │  step()
         │  ────────────────>  │  ───────────────────> │
         │                     │                       │
         │  async generator()  │  process_outputs()    │  outputs
         │  <────────────────  │  <─────────────────── │
         │                     │                       │
   AsyncStream.generator()   Route to stream         Create RequestOutput

Reference: AsyncStream + RequestTracker from vllm/engine/async_llm_engine.py,
           Simplified for baby-vllm by removing complex features like
           multi-engine, beam search, and multi-step.
"""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Optional, Union

from babyvllm.engine.outputs import RequestOutput

# ---------------------------------------------------------------------------
# STOP_ITERATION: Stream termination sentinel
#
# Why not use None?
#   None could be a valid return value (although we don't directly yield None),
#   using a custom sentinel clearly distinguishes "normal stream end" from "unexpected None".
#   Reference: asyncio.Queue commonly uses sentinel pattern in Python standard library.
#
# Why a plain object (not an Exception instance)?
#   STOP_ITERATION must NOT pass the _is_raisable check — it is a sentinel
#   that tells the consumer to exit normally via StopAsyncIteration.
#   If it were a BaseException instance, the generator would raise it instead.
# ---------------------------------------------------------------------------
STOP_ITERATION = object()  # Sentinel, not a real exception


# ===========================================================================
# AsyncStream —— Per-request async output channel
# ===========================================================================

class AsyncStream:
    """
    Per-request async output stream, wrapping an asyncio.Queue.

    Each online request corresponds to one AsyncStream instance.
    RequestOutput produced by engine step() is put into the queue via put().
    The API layer consumes it asynchronously via generator() for streaming output.

    Attributes:
        request_id: The associated request ID (equals Sequence.seq_id).

    Lifecycle:
        ┌─────────┐   put()    ┌──────────┐   put()    ┌─────────┐
        │ RUNNING │──────────->│ RUNNING  │───────────>│ FINISHED│
        │ (empty) │            │(has data)│            │  (done) │
        └─────────┘            └──────────┘            └─────────┘
                                   │         finish()       ▲
                                   └────────────────────────┘

    Usage Example:
        stream = AsyncStream(request_id=1)
        # Producer (engine side)
        stream.put(RequestOutput(request_id=1, text="Hello", ...))
        stream.finish()
        # Consumer (API side)
        async for output in stream.generator():
            print(output.text)  # → "Hello"
    """

    def __init__(self, request_id: int, cancel: callable = None) -> None:
        """
        Args:
            request_id: Unique request ID for logging and debugging.
            cancel: Optional callback hook when generator is externally cancelled
                    (GeneratorExit), typically RequestTracker.abort_request to ensure
                    the engine side also cleans up resources.
        """
        self.request_id = request_id
        self._cancel = cancel
        # asyncio.Queue is thread-safe async queue with FIFO semantics.
        # No maxsize parameter means unlimited capacity, prevents blocking when engine puts to queue.
        self._queue: asyncio.Queue = asyncio.Queue()
        self._finished = False

    def put(self, item: Union[RequestOutput, Exception]) -> None:
        """
        Put RequestOutput into the queue (non-blocking).

        Called by the engine side to push newly generated tokens to the stream after each step.
        If the stream is already finished (e.g., request cancelled), the item is silently discarded.

        Args:
            item: RequestOutput object, or Exception in error scenarios.
        """
        if not self._finished:
            self._queue.put_nowait(item)

    def finish(
        self,
        exception: Optional[Union[BaseException, type[BaseException]]] = None,
    ) -> None:
        """
        Mark the stream as finished.

        After calling this method, no more put() calls are accepted.
        Puts STOP_ITERATION sentinel at the end of the queue.
        Generator will exit normally when it consumes the sentinel.

        Args:
            exception: If provided, generator will raise this exception instead of
                       exiting normally. Used to propagate engine errors or request
                       cancellation signals.
        """
        if not self._finished:
            self._finished = True
            # If exception instance or class is provided, put it in queue.
            # Otherwise put STOP_ITERATION sentinel for normal termination.
            self._queue.put_nowait(
                exception if self._is_raisable(exception) else STOP_ITERATION
            )

    @property
    def finished(self) -> bool:
        """Whether the stream has been marked as finished."""
        return self._finished

    def generator(self):
        """
        Return an async iterator that yields RequestOutput items.

        API layer consumes output via `async for output in stream.generator()`.
        Iterator exits normally when STOP_ITERATION sentinel appears in the queue.
        Iterator raises exception when exception object appears in the queue.

        Cancellation Handling:
          If the caller (API layer) disconnects during iteration,
          Python's async for calls aclose() on the iterator, which triggers
          the _cancel hook to ensure the engine side also releases resources.

        Yields:
            RequestOutput: Output chunk from each step.

        Raises:
            asyncio.CancelledError: When generator is externally cancelled.

        Usage Example:
            async for output in stream.generator():
                yield f"data: {output.text}\\n\\n"
            yield "data: [DONE]\\n\\n"
        """
        return _AsyncStreamIter(self)

    @staticmethod
    def _is_raisable(value):
        """
        Check if a value is a "raisable" exception object.

        Includes:
          - Exception instances (e.g., CancelledError(), ValueError("msg"))
          - Exception classes (e.g., ValueError) - rarely used
        """
        return isinstance(value, BaseException) or (
            isinstance(value, type) and issubclass(value, BaseException)
        )


class _AsyncStreamIter:
    """
    Custom async iterator wrapping an AsyncStream's internal queue.

    Uses __anext__ / aclose protocol instead of native async generator
    (async def / yield) so that Python's async for reliably calls aclose()
    when the consumer disconnects, including for GeneratorExit (BaseException).
    Native async generators skip the GeneratorExit handler in some Python versions.
    """

    def __init__(self, stream: AsyncStream) -> None:
        self._stream = stream
        self._exhausted = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._exhausted:
            raise StopAsyncIteration
        result = await self._stream._queue.get()
        if result is STOP_ITERATION:
            self._exhausted = True
            raise StopAsyncIteration
        if self._stream._is_raisable(result):
            self._exhausted = True
            raise result
        return result

    async def aclose(self):
        if not self._exhausted:
            self._exhausted = True
            if self._stream._cancel is not None:
                self._stream._cancel(self._stream.request_id)
                self._stream._cancel = None

    def __del__(self):
        if not self._exhausted and self._stream._cancel is not None:
            self._stream._cancel(self._stream.request_id)
            self._stream._cancel = None


# ===========================================================================
# RequestTracker —— Global request tracker
# ===========================================================================

class RequestTracker:
    """
    Synchronous request tracking abstraction that manages AsyncStreams for all in-flight requests.

    Acts as intermediary between engine and API layer in AsyncLLMEngine:
      - API layer registers new requests via add_request() and receives AsyncStream
      - Engine loop gets pending requests via get_new_and_aborted_requests()
      - Engine loop routes outputs via process_request_output() to corresponding streams

    Attributes:
        _request_streams: {request_id: AsyncStream} mapping for active requests.
                          Removed from mapping when request completes.
        _new_requests: Queue of requests waiting to be processed by engine.
        _aborted_requests: Queue of request IDs waiting to be cancelled by engine.
        new_requests_event: asyncio.Event set when new requests arrive,
                            engine loop waits on this event to sleep during idle periods.

    Usage Example (Simplified Engine Loop):
        tracker = RequestTracker()

        # --- API Thread ---
        stream = tracker.add_request(request_id=1,
                                     prompt_token_ids=[1,2,3],
                                     sampling_params=...)
        # Returns stream, API layer waits on generator()

        # --- Engine Loop ---
        new_reqs, aborted = tracker.get_new_and_aborted_requests()
        # new_reqs = [{"request_id": 1, "prompt_token_ids": [1,2,3], ...}]
        for req in new_reqs:
            engine.add_request(...)  # Add request to scheduler

        # After engine step...
        output = RequestOutput(request_id=1, text="Hello", ...)
        tracker.process_request_output(output)
        # output is routed to AsyncStream for request_id=1
    """

    def __init__(self) -> None:
        # Active requests: moved from _new_requests here after claimed by engine
        self._request_streams: dict[int, AsyncStream] = {}
        # Pending new requests queue: elements are (AsyncStream, dict) tuples
        # dict contains {"request_id": int, "prompt_token_ids": [...], ...}
        self._new_requests: asyncio.Queue = asyncio.Queue()
        # Track request_ids that are still in _new_requests (not yet claimed).
        # asyncio.Queue doesn't support O(1) membership test, so we mirror the ids here.
        self._pending_request_ids: set[int] = set()
        # Pending cancellation request IDs queue
        self._aborted_requests: asyncio.Queue = asyncio.Queue()
        # Event: set() when new requests arrive, engine loop waits on this event for idle sleep
        self.new_requests_event = asyncio.Event()

    # ---- Query Methods ----

    def __len__(self) -> int:
        """Number of active requests."""
        return len(self._request_streams)

    def has_new_requests(self) -> bool:
        """
        Whether there are unprocessed new requests.
        Used by engine loop to decide if immediate processing is needed.
        """
        return not self._new_requests.empty()

    async def wait_for_new_requests(self):
        """
        Wait for new requests to arrive.
        Called by engine loop during idle periods to avoid busy-waiting.

        If there are already new requests (queue not empty), returns immediately;
        otherwise blocks until add_request() calls new_requests_event.set().
        """
        if not self.has_new_requests():
            await self.new_requests_event.wait()
        # Clear event flag for next wait cycle
        self.new_requests_event.clear()

    # ---- Request Lifecycle Management ----

    def add_request(
        self,
        request_id: int,
        prompt_token_ids: list[int],
        sampling_params=None,
        **extra_kwargs,
    ) -> AsyncStream:
        """
        Register a new request.

        Called by API layer.
        Creates an AsyncStream and puts it in the new requests queue,
        waiting for the engine loop to claim it in the next iteration.

        Args:
            request_id: Unique request ID. Caller is responsible for ensuring global uniqueness
                        (typically auto-generated using Sequence.counter).
            prompt_token_ids: List of token IDs for the prompt.
                              Later used to populate prompt_token_ids field when creating RequestOutput.
            sampling_params: Sampling parameters (used by AsyncLLMEngine).
            **extra_kwargs: Additional parameters passed directly to engine's add_request().

        Returns:
            AsyncStream: API layer consumes output via stream.generator().

        Raises:
            KeyError: If request_id already exists.

        Example:
            stream = tracker.add_request(
                request_id=1,
                prompt_token_ids=tokenizer.encode("Hello, world!"),
                sampling_params=SamplingParams(max_tokens=64),
            )
            # In another coroutine:
            async for output in stream.generator():
                print(f"Request {output.request_id}: {output.text}")
        """
        if request_id in self._request_streams or request_id in self._pending_request_ids:
            raise KeyError(f"Request {request_id} already exists.")

        # Create abort hook: automatically call tracker.abort_request when stream is cancelled.
        # Use lambda with default arg to avoid late-binding closure issues.
        abort_callback = lambda rid: self.abort_request(request_id=rid)

        stream = AsyncStream(request_id, cancel=abort_callback)
        self._new_requests.put_nowait((stream, {
            "request_id": request_id,
            "prompt_token_ids": prompt_token_ids,
            "sampling_params": sampling_params,
            **extra_kwargs,
        }))
        self._pending_request_ids.add(request_id)

        # Wake up engine loop if it's waiting
        self.new_requests_event.set()

        return stream

    def abort_request(
        self,
        request_id: int,
        *,
        exception: Optional[Union[BaseException, type[BaseException]]] = None,
    ) -> None:
        """
        Cancel a request.

        If the request is still active, remove it from _request_streams and finish its stream.
        Also put request_id in _aborted_requests queue, engine loop will remove it from scheduler.

        Args:
            request_id: ID of the request to cancel.
            exception: Optional exception to propagate to stream.generator() consumer.
                       Default None means normal cancellation (generator raises CancelledError).

        Example:
            # Client disconnects → API layer cancels request
            tracker.abort_request(request_id=1, exception=asyncio.CancelledError())
        """
        self._aborted_requests.put_nowait(request_id)

        # If request is still pending (not yet claimed), remove it
        self._pending_request_ids.discard(request_id)

        # If request is still active, immediately finish its stream
        stream = self._request_streams.pop(request_id, None)
        if stream is not None:
            stream.finish(exception=exception)

    def process_exception(
        self,
        request_id: int,
        exception: BaseException,
    ) -> None:
        """
        Propagate an engine-level exception to a specific request's stream.

        Delegates to abort_request with the exception. The stream consumer
        (async generator) will raise the exception when it reads from the queue.

        Why this exists:
          When _add_request_to_engine() fails for a single request (e.g., invalid
          sampling params, resource allocation failure), we want to notify only
          that request's client. Other concurrent requests should continue
          unaffected. This is the "Level 1" (per-request) error handling.

        How it works:
          abort_request() does two things:
            1. Puts request_id in _aborted_requests queue (engine loop will
               clean up scheduler state on next iteration)
            2. Finishes the AsyncStream with the given exception, which causes
               the consumer's async for loop to raise it

        Args:
            request_id: Internal engine request ID to fail.
            exception: The exception to propagate (e.g., ValueError, TypeError).

        Example:
            try:
                engine._add_request_to_engine(request_data)
            except ValueError as e:
                tracker.process_exception(request_data["request_id"], e)
                # The API layer's async for loop will raise ValueError
                # Other concurrent requests continue unaffected
        """
        self.abort_request(request_id, exception=exception)

    def process_exception_all(
        self,
        exception: BaseException,
    ) -> None:
        """
        Propagate an engine-level exception to ALL active request streams.

        Called when a catastrophic engine failure occurs (e.g., GPU OOM,
        CUDA error, multiprocessing failure). Since the engine operates on
        all sequences as a single batch, a failure affects all in-flight
        requests simultaneously.

        Why this exists:
          This is the "Level 2" (catastrophic) error handler. Unlike
          process_exception() which targets a single request, this method
          fans out the exception to every active stream. All clients will
          see the same exception in their async for loop.

        Design note — tuple() snapshot:
          Uses tuple(self._request_streams.keys()) to create a snapshot of
          request IDs before iterating. This is necessary because
          abort_request() internally calls _request_streams.pop(request_id),
          which mutates the dict during iteration. Without the snapshot,
          the iteration would raise RuntimeError: dictionary changed size.

        Args:
            exception: The exception to propagate to all request streams.
                       (e.g., torch.cuda.OutOfMemoryError, RuntimeError)

        Example:
            try:
                engine.step()
            except torch.cuda.OutOfMemoryError as e:
                tracker.process_exception_all(e)
                # All clients' async for loops will raise OutOfMemoryError
                # All scheduler sequences cleaned up separately
        """
        for request_id in tuple(self._request_streams.keys()):
            self.abort_request(request_id, exception=exception)

    def get_new_and_aborted_requests(self) -> tuple[list[dict], set[int]]:
        """
        Get pending new requests and requests to cancel.

        Called by engine loop at the start of each iteration:
          1. First collect all request IDs to cancel
          2. Then process new requests queue:
             - If new request ID is already in cancellation list (extreme race), reject directly
             - Otherwise register its stream to _request_streams and return

        Returns:
            (new_requests, aborted_request_ids):
              - new_requests: list of dict, each containing request_id,
                              prompt_token_ids, sampling_params, etc.
              - aborted_request_ids: set of int, requests that engine needs to remove from scheduler.

        Example:
            new_reqs, aborted = tracker.get_new_and_aborted_requests()
            # new_reqs = [{"request_id": 1, "prompt_token_ids": [...], ...}]
            # aborted = {3, 5}
            for req in new_reqs:
                engine.add_request(**req)
            for rid in aborted:
                engine.abort_request(rid)
        """
        # Step 1: Collect requests to cancel
        aborted_request_ids: set[int] = set()
        while not self._aborted_requests.empty():
            aborted_request_ids.add(self._aborted_requests.get_nowait())

        # Step 2: Process new requests
        new_requests: list[dict] = []
        while not self._new_requests.empty():
            stream, request_data = self._new_requests.get_nowait()
            request_id = stream.request_id
            self._pending_request_ids.discard(request_id)

            # Edge case: request cancelled immediately after being added to new queue
            if request_id in aborted_request_ids:
                stream.finish(asyncio.CancelledError)
            else:
                # Register as active request
                self._request_streams[request_id] = stream
                new_requests.append(request_data)

        return new_requests, aborted_request_ids

    def get_new_requests(self) -> list[dict]:
        """
        Get only new requests (does not process cancellation queue).

        Difference from get_new_and_aborted_requests():
          - get_new_and_aborted_requests() processes both queues
          - get_new_requests() only drains _new_requests queue, does not touch _aborted_requests

        Returns:
            list of dict: List of new request data, each containing request_id,
                          prompt_token_ids, sampling_params, etc.

        Example:
            new_reqs = tracker.get_new_requests()
            for req in new_reqs:
                engine.add_request(**req)
        """
        new_requests: list[dict] = []
        while not self._new_requests.empty():
            stream, request_data = self._new_requests.get_nowait()
            self._pending_request_ids.discard(stream.request_id)
            # Register as active request
            self._request_streams[stream.request_id] = stream
            new_requests.append(request_data)
        return new_requests

    def process_step_outputs(
        self,
        request_outputs: list[RequestOutput],
    ) -> None:
        """
        Process all RequestOutputs produced by one engine step in batch.

        This is the batch version of process_request_output(),
        called by engine loop after each step() to route the entire batch of outputs at once.

        Args:
            request_outputs: List of all RequestOutputs produced in current step.

        Internal Implementation:
          Calls process_request_output() for each RequestOutput:
            - Unfinished request → put to corresponding stream
            - Finished request → put + finish stream + remove from active list

        Example:
            # In AsyncLLMEngine._engine_step():
            outputs, is_prefill = engine.step()
            request_outputs = []
            for seq_id, token_ids in outputs:
                request_outputs.append(RequestOutput(
                    request_id=seq_id,
                    text=tokenizer.decode(token_ids),
                    token_ids=token_ids,
                    finished=True, # Current engine.step() only returns completed sequences
                    prompt_token_ids=prompt_map[seq_id],
                ))
            tracker.process_step_outputs(request_outputs)
        """
        for output in request_outputs:
            self.process_request_output(output)

    def process_request_output(
        self,
        request_output: RequestOutput,
    ) -> None:
        """
        Route RequestOutput produced by engine to the corresponding request's AsyncStream.

        Called by engine loop after each step() for each request that made progress.
        If the request is completed (request_output.finished=True), also finish its stream
        and remove from active list.

        Args:
            request_output: Output chunk for a single request.

        Silent Handling:
          - If request_id is not in _request_streams (e.g., request was cancelled),
            output is silently discarded.
            This is normal as there's a race window between abort and step.

        Example:
            for seq_id, token_ids, finished in step_outputs:
                output = RequestOutput(
                    request_id=seq_id,
                    text=tokenizer.decode(token_ids),
                    token_ids=token_ids,
                    finished=finished,
                    prompt_token_ids=prompt_map[seq_id],
                )
                tracker.process_request_output(output)
        """
        request_id = request_output.request_id

        if request_output.finished:
            # Request completed → retrieve and remove stream.
            stream = self._request_streams.pop(request_id, None)
        else:
            # Request still running → get stream (don't remove).
            stream = self._request_streams.get(request_id)

        # If stream doesn't exist (may have been aborted), silently discard.
        if stream is not None:
            stream.put(request_output)
            if request_output.finished:
                stream.finish()
