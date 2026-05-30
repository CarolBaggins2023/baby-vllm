"""
baby-vllm OpenAI-Compatible REST API Server
=============================================

Core Responsibilities:
  Expose an HTTP API matching the OpenAI /v1/completions and /v1/chat/completions
  endpoints. Wraps AsyncLLMEngine behind Pydantic-validated request/response models
  and supports both streaming (SSE, Server-Sent Events) and non-streaming modes.

=============================================================
Architecture Overview
=============================================================

  Client (curl/openai SDK)
       │
       │  POST /v1/completions   {"model":"qwen","prompt":"Hello","max_tokens":32,...}
       │  ────────────────────────────────────────────────────────────────────────>
       │
       │  ┌──────────────────────────────────────────────────────────────────┐
       │  │                    FastAPI Application                           │
       │  │                                                                  │
       │  │  1. Pydantic validates request body into CompletionRequest       │
       │  │  2. Map request fields to SamplingParams (clamp temperature!)    │
       │  │  3. Generate string request_id (not engine's internal int ID)    │
       │  │  4. Call _engine.generate(prompt, sampling_params)               │
       │  │  5. Async iterate RequestOutput from generator                   │
       │  │  6. Map RequestOutput fields to CompletionResponse or SSE chunks │
       │  │  7. Return JSON response or StreamingResponse                    │
       │  └─────────────┬────────────────────────────────────────────────────┘
       │                │
       │                │  _engine.generate()
       │                │  (async generator)
       │                ▼
       │  ┌──────────────────────────────────────────────────────────────────┐
       │  │                    AsyncLLMEngine                                │
       │  │  (background loop, scheduler, model runner)                      │
       │  └──────────────────────────────────────────────────────────────────┘
       │
       │  JSON Response or SSE Stream
       │  <────────────────────────────────────────────────────────────────────────

  Module-level State:
    _engine (AsyncLLMEngine | None) — Singleton engine instance, injected by CLI launcher
      before uvicorn.run(). All endpoints access this shared engine.
    _model_name (str | None) — Cached model name for response headers, computed lazily.

  Lifecycle:
    1. CLI launcher (cli.py) creates AsyncLLMEngine and sets api_server._engine
    2. uvicorn.run(api_server.app) starts HTTP server
    3. FastAPI lifespan: on startup, nothing (engine already created); on shutdown, await _engine.stop()
    4. Each request handler accesses module-level _engine
"""

# ===========================================================================
# Imports
# ===========================================================================

from __future__ import annotations

import asyncio  # For asyncio.CancelledError handling in streaming
import json     # For model_dump_json() on Pydantic models
import os       # For path basename in _get_model_name()
import time     # For created Unix timestamps in responses
import uuid     # For generating request IDs (CR-3)
from contextlib import asynccontextmanager  # For FastAPI lifespan context manager
from typing import (  # Type annotations for function signatures
    Any,
    AsyncGenerator,
    Optional,
    Union,
)

# FastAPI framework imports
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware  # CORS support (MI-3)
from fastapi.responses import Response, StreamingResponse  # HTTP response types

# Pydantic v2 imports for request/response validation
from pydantic import BaseModel, field_validator

# baby-vllm internal imports
from babyvllm.engine.async_llm_engine import AsyncLLMEngine
from babyvllm.engine.outputs import RequestOutput
from babyvllm.sampling_params import SamplingParams

# ===========================================================================
# Module-Level State (Singleton Pattern)
# ===========================================================================
#
# Design Rationale:
#   AsyncLLMEngine must be a singleton because:
#   1. The underlying LLMEngine holds GPU resources (model weights, KV cache).
#   2. The scheduler manages all sequences in a shared queue.
#   3. Multiprocessing workers are bound to the main process.
#   Creating multiple engines would cause GPU OOM and IPC conflicts.
#
# The CLI launcher (cli.py) sets _engine before calling uvicorn.run().
# All endpoint handlers access this shared engine instance.
# ===========================================================================

_engine: Optional[AsyncLLMEngine] = None
"""Singleton AsyncLLMEngine instance, injected by CLI launcher before server starts."""

_model_name: Optional[str] = None
"""Cached model name for response headers. Computed lazily via _get_model_name()."""

# ===========================================================================
# Utility Functions
# ===========================================================================


def _get_model_name() -> str:
    """
    Get the model name for API responses.

    Resolution Priority:
      1. Use HuggingFace config's name_or_path (basename only) — reflects the model repo name
      2. Fall back to the filesystem directory name (basename of model path)

    Why not just use the model directory basename directly?
      - HuggingFace configs store the original model identifier (e.g., "Qwen/Qwen2-0.5B-Instruct"),
        which is more meaningful than a local path like "/home/user/models/qwen"
      - The basename extraction strips repository prefixes for cleaner output

    Integration:
      Called by every endpoint handler that needs to populate the "model" field in responses.

    Returns:
        str: A human-readable model name (e.g., "Qwen2-0.5B-Instruct", or "unknown" if engine is not set)

    Example:
        >>> _engine = AsyncLLMEngine("/data/models/Qwen2-0.5B-Instruct")
        >>> _get_model_name()
        "Qwen2-0.5B-Instruct"  # If HF config has name_or_path set
    """
    global _model_name

    # Return cached value if already computed. Model name never changes during server lifetime.
    if _model_name is not None:
        return _model_name

    if _engine is None:
        return "unknown"

    # Try HuggingFace config's name_or_path first
    # hf_config is AutoConfig | None (from Config.__post_init__)
    hf_config = _engine.engine.config.hf_config
    if hf_config is not None and hasattr(hf_config, "name_or_path") and hf_config.name_or_path:
        # name_or_path typically looks like "Qwen/Qwen2-0.5B-Instruct"
        # os.path.basename extracts "Qwen2-0.5B-Instruct"
        _model_name = os.path.basename(hf_config.name_or_path)
        return _model_name

    # Fallback: use filesystem directory name
    # config.model is the raw path passed to --model CLI argument
    # rstrip("/\\") handles trailing slashes on Windows and Unix paths
    _model_name = os.path.basename(_engine.engine.config.model.rstrip("/\\"))
    return _model_name


def _random_id(prefix: str = "cmpl") -> str:
    """
    Generate a UUID-based request ID string for API responses (CR-3).

    Why not use engine's internal int request_id?
      - OpenAI API expects string IDs like "cmpl-{uuid}" or "chatcmpl-{uuid}"
      - Engine's internal IDs are auto-increment integers (0, 1, 2, ...)
      - We keep the API layer's ID space completely separate from the engine layer

    Design:
      Uses uuid.uuid4() for randomness and uniqueness.
      Takes the first 8 hex characters (2^32 possibilities) — sufficient for
      a single-server deployment.

    Args:
        prefix: ID prefix ("cmpl" for completions, "chatcmpl" for chat completions)

    Returns:
        str: A unique request ID string, e.g., "cmpl-a1b2c3d4"

    Example:
        >>> _random_id("cmpl")
        "cmpl-e8f3a91c"
        >>> _random_id("chatcmpl")
        "chatcmpl-7b2d4f6a"
    """
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _normalize_prompt(prompt: Union[str, list[int]]) -> Union[str, list[int]]:
    """
    Normalize the prompt input. Currently a pass-through.

    Validation is handled by Pydantic's @field_validator('prompt') on CompletionRequest,
    which rejects list[str] and list[list[int]]. This function exists as an extension
    point for future normalization logic (e.g., truncation warnings, encoding checks).

    Args:
        prompt: Raw prompt (str or list[int]) from the validated request.

    Returns:
        Union[str, list[int]]: The same prompt, unchanged.
    """
    return prompt


def _clamp_temperature(temp: Optional[float]) -> float:
    """
    Clamp temperature to avoid SamplingParams assertion failure (CR-1).

    Problem:
      SamplingParams.__post_init__ asserts: assert self.temperature > 1e-10
      This means temperature=0 (greedy sampling, valid in OpenAI API) would raise
      an AssertionError at the engine layer.

    Solution:
      Clamp temperature to a minimum of 1e-6 (which passes the > 1e-10 check).
      This is effectively greedy sampling but avoids the assertion.

    Why not modify SamplingParams itself?
      SamplingParams is Phase 1 code. The API layer handles the compatibility
      mapping. This follows the Adapter pattern — the API layer adapts between
      OpenAI's spec and baby-vllm's internal constraints.

    Args:
        temp: Temperature value from the client request (None defaults to 1.0).

    Returns:
        float: Clamped temperature value, guaranteed to be >= 1e-6.

    Example:
        >>> _clamp_temperature(0.0)
        1e-06
        >>> _clamp_temperature(0.7)
        0.7
        >>> _clamp_temperature(None)
        1.0
    """
    if temp is None:
        return 1.0
    return max(temp, 1e-6)


# ===========================================================================
# Pydantic Models — Base
# ===========================================================================


class OpenAIBaseModel(BaseModel):
    """
    Base model for all OpenAI-compatible request/response models.

    Key Design Decision: extra="allow"
      OpenAI API clients may send fields that baby-vllm doesn't implement yet
      (e.g., "logprobs", "top_p", "seed", "tools", "response_format").
      Using extra="allow" (instead of "forbid") means we silently ignore unknown
      fields rather than rejecting the entire request. This maximizes client
      compatibility.

      Configuration:
        Uses model_config dict (Pydantic v2 style) rather than ConfigDict class,
        for simplicity and clarity.

    Example:
        # Client sends: {"model": "x", "prompt": "Hi", "unknown_field": 42}
        # Request is accepted; "unknown_field" is silently ignored.
    """
    model_config = {"extra": "allow"}


# ===========================================================================
# Pydantic Models — UsageInfo
# ===========================================================================


class UsageInfo(BaseModel):
    """
    OpenAI-compatible token usage statistics (CR-4).

    Computed in the API layer from RequestOutput fields:
      - prompt_tokens = len(output.prompt_token_ids)
      - completion_tokens = len(output.token_ids)  (generated tokens only)
      - total_tokens = prompt_tokens + completion_tokens

    Why compute in API layer rather than engine layer?
      RequestOutput doesn't have explicit prompt_tokens/completion_tokens fields.
      We derive them from the raw token_ids arrays, which are always available.

    All fields default to 0 for safety (e.g., if computation fails mid-request).
    """
    prompt_tokens: int = 0
    """Number of tokens in the input prompt."""

    total_tokens: int = 0
    """Total tokens consumed (prompt + generated)."""

    completion_tokens: int = 0
    """Number of tokens generated by the model."""


class MetricsInfo(BaseModel):
    """
    Per-request performance timing metrics.

    Populated from RequestOutput fields when a request completes (finished=True).
    These are collected by AsyncLLMEngine and passed through to the API response.

    In the current engine (which only reports completed sequences), TTFT is
    approximated as the end-to-end latency. These metrics will become precise
    when per-step incremental output is supported in the engine.
    """
    ttft: Optional[float] = None
    """Time to first token in seconds (approximate in current engine)."""

    tpot: Optional[float] = None
    """Time per output token in seconds (approximate in current engine)."""

    total_time: Optional[float] = None
    """Total inference time in seconds for this request."""


# ===========================================================================
# Pydantic Models — /v1/completions
# ===========================================================================


class CompletionRequest(OpenAIBaseModel):
    """
    Request body for POST /v1/completions (OpenAI-compatible).

    Field Mapping to baby-vllm:
      prompt         → AsyncLLMEngine.generate(prompt=...)
      temperature    → SamplingParams(temperature=...), clamped to >= 1e-6 (CR-1)
      max_tokens     → SamplingParams(max_tokens=...), default 16
      ignore_eos     → SamplingParams(ignore_eos=...)
      stream         → Controls StreamingResponse vs JSONResponse
      echo           → Prepends prompt text to completion (API layer)
      stop           → Accepted but NOT enforced in Phase 4
      n              → Only n=1 is supported (single completion per request)

    Why default max_tokens=16 instead of SamplingParams default of 64?
      OpenAI API defaults max_tokens to 16. We follow the OpenAI spec at the API
      boundary, even though the engine's default is higher.

    Example:
        POST /v1/completions
        {
          "model": "qwen",
          "prompt": "Once upon a time",
          "max_tokens": 32,
          "temperature": 0.7,
          "stream": false
        }
    """
    model: Optional[str] = None
    """Model identifier. Currently unused (single-model deployment), but accepted for OpenAI SDK compatibility."""

    prompt: Union[str, list[int]]
    """
    The prompt to generate completions for.
    Accepted types:
      - str: Natural language prompt, tokenized by the engine
      - list[int]: Pre-tokenized prompt as integer token IDs
    Rejected types (validated by @field_validator):
      - list[str]: Batch of prompts (not supported)
      - list[list[int]]: Batch of token ID lists (not supported)
    """

    max_tokens: Optional[int] = 16
    """
    Maximum number of tokens to generate.
    Default 16 matches OpenAI API default.
    Maps to SamplingParams.max_tokens.
    """

    temperature: Optional[float] = 1.0
    """
    Sampling temperature (0.0 to 2.0 in OpenAI spec).
    CLAMPED to minimum 1e-6 before constructing SamplingParams (CR-1).
    Default 1.0 matches OpenAI API.
    """

    stream: Optional[bool] = False
    """
    Whether to stream results as SSE (Server-Sent Events).
    False → JSON response (non-streaming)
    True  → text/event-stream response (streaming)
    """

    stop: Optional[Union[str, list[str]]] = None
    """
    Stop sequences that terminate generation early.
    ACCEPTED but NOT enforced in Phase 4. Included for API compatibility.
    Will be implemented in a future phase.
    """

    echo: Optional[bool] = False
    """
    Whether to include the prompt text in the completion output.
    When True, the response text = prompt + generated text.
    Echo is applied at the API layer (not engine layer) by prepending
    decoded prompt tokens.
    Example:
      prompt="Hello, my name is"
      generated=" Bob"
      echo=False → text=" Bob"
      echo=True  → text="Hello, my name is Bob"
    """

    n: int = 1
    """
    Number of completions to generate per prompt.
    Only n=1 is supported in Phase 4. Values > 1 will raise a validation error.
    Maps to single engine.generate() call.
    """

    ignore_eos: bool = False
    """
    Whether to ignore the EOS (end-of-sequence) token during generation.
    Maps to SamplingParams.ignore_eos.
    When True, the model continues generating past EOS.
    """

    user: Optional[str] = None
    """
    User identifier for abuse monitoring (OpenAI API field).
    Accepted but currently unused in baby-vllm.
    """

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v):
        """
        Validate the prompt field — only accept str and list[int] (CR-2).

        What is rejected:
          - list[str]: e.g., ["prompt1", "prompt2"] — batch of string prompts
          - list[list[int]]: e.g., [[1,2,3], [4,5,6]] — batch of token ID lists

        Why reject these?
          AsyncLLMEngine.generate() signature is:
            generate(prompt: Union[str, list[int]], ...)
          It does NOT support list[str] or list[list[int]]. Accepting these
          would cause a TypeError deep in the engine call stack, which is harder
          to debug than a clean Pydantic validation error at the API boundary.

        Design Pattern: Fail Fast
          Validate at the outermost layer (API request parsing) rather than
          deep in the engine. This gives clear error messages to clients.

        Args:
            v: The prompt value from the request body.

        Returns:
            The original prompt value (passed through if valid).

        Raises:
            ValueError: If prompt is a list containing non-integers.
        """
        if isinstance(v, list):
            # Review fix: Check for empty list (vacuous truth in all() check below).
            # Empty prompt would create a zero-token sequence, producing confusing output.
            if len(v) == 0:
                raise ValueError(
                    "prompt list must not be empty. "
                    "Provide at least one token ID, e.g., [15496]."
                )
            if not all(isinstance(x, int) for x in v):
                raise ValueError(
                    "prompt list must contain only integers (list[int]). "
                    "list[str] and list[list[int]] are not supported. "
                    "Use a single string or a single list of token IDs."
                )
        return v

    @field_validator("n")
    @classmethod
    def validate_n(cls, v):
        """
        Validate the n parameter — only n=1 is supported.

        Why only n=1?
          n > 1 requires beam search or multiple independent sampling runs per request.
          The current engine only supports one sequence per generate() call.
          Supporting n > 1 would require engine-level changes to the scheduler
          and sequence management.

        Args:
            v: The n value from the request.

        Returns:
            The original value if valid.

        Raises:
            ValueError: If n > 1.
        """
        if v > 1:
            raise ValueError(
                "n > 1 is not supported yet. "
                "Multiple completions per request require beam search "
                "or parallel sampling infrastructure not yet implemented."
            )
        return v


class CompletionResponseChoice(BaseModel):
    """
    A single completion choice within a CompletionResponse.

    Matches OpenAI's response format:
      {
        "index": 0,
        "text": " generated text here",
        "finish_reason": "stop",
        "logprobs": null
      }
    """
    index: int = 0
    """Index of this choice in the choices array. Always 0 (single completion)."""

    text: str = ""
    """The generated text (or full text with echo). Set to empty string during prefill."""

    finish_reason: Optional[str] = None
    """
    Reason for completion termination.
    Values: "stop" (EOS token generated), "length" (max_tokens reached).
    None during streaming (incomplete chunks).
    In Phase 4: always "stop" when finished (engine only outputs completed sequences).
    """

    logprobs: Optional[Any] = None
    """
    Log probabilities for generated tokens.
    Always None in Phase 4 (logprob computation not yet implemented).
    Included for API compatibility with OpenAI SDK.
    """


class CompletionResponse(OpenAIBaseModel):
    """
    Full response body for POST /v1/completions (non-streaming).

    Matches OpenAI's response format:
      {
        "id": "cmpl-a1b2c3d4",
        "object": "text_completion",
        "created": 1712345678,
        "model": "Qwen2-0.5B-Instruct",
        "choices": [...],
        "usage": {"prompt_tokens": 5, "completion_tokens": 32, "total_tokens": 37},
        "metrics": {"ttft": 0.05, "tpot": 0.01, "total_time": 1.23}
      }
    """
    id: str
    """Unique request ID, e.g., "cmpl-a1b2c3d4". Generated by _random_id() (CR-3)."""

    object: str = "text_completion"
    """Object type identifier per OpenAI API spec. Always "text_completion"."""

    created: int
    """Unix timestamp (seconds) when the response was created."""

    model: str
    """Model identifier, e.g., "Qwen2-0.5B-Instruct". From _get_model_name()."""

    choices: list[CompletionResponseChoice]
    """Array of completion choices. In Phase 4, always a single-element list."""

    usage: UsageInfo
    """Token usage statistics computed from RequestOutput (CR-4)."""

    metrics: Optional[MetricsInfo] = None
    """Per-request performance timing metrics (baby-vllm extension)."""


class CompletionStreamResponse(OpenAIBaseModel):
    """
    A single SSE chunk in a streaming /v1/completions response.

    Each chunk matches OpenAI's streaming format:
      data: {"id":"cmpl-...","object":"text_completion","created":...,"model":"...","choices":[...]}

    Usage and metrics are only included in the FINAL chunk (when finish_reason is set).
    """
    id: str
    """Same request_id across all chunks of the same request."""

    object: str = "text_completion"
    """Always "text_completion"."""

    created: int
    """Same created timestamp across all chunks of the same request."""

    model: str
    """Same model name across all chunks."""

    choices: list[CompletionResponseChoice]
    """Completion choices for this chunk."""

    usage: Optional[UsageInfo] = None
    """Token usage statistics (only in the final chunk)."""

    metrics: Optional[MetricsInfo] = None
    """Per-request performance timing metrics (only in the final chunk)."""


# ===========================================================================
# Pydantic Models — /v1/chat/completions
# ===========================================================================


class ChatMessage(BaseModel):
    """
    A single message in a chat conversation.

    Matches OpenAI's ChatML format:
      {"role": "system", "content": "You are a helpful assistant."}
      {"role": "user", "content": "Hello!"}
      {"role": "assistant", "content": "Hi! How can I help you?"}

    Used in both request (ChatCompletionRequest.messages) and response
    (ChatCompletionResponseChoice.message).
    """
    role: str
    """
    Message role. Standard values:
      - "system": System-level instructions (persona, behavior rules)
      - "user": End-user message
      - "assistant": Model's response
    """

    content: str
    """Message content text. Always a string (multimodal not yet supported)."""


class ChatCompletionRequest(OpenAIBaseModel):
    """
    Request body for POST /v1/chat/completions (OpenAI-compatible).

    Message → Prompt Conversion:
      Chat messages are converted to a flat token ID list before being sent to
      the engine. Two conversion strategies:
        1. Chat Template (preferred): Uses tokenizer.apply_chat_template() with
           the model's built-in Jinja2 chat template. This adds special tokens
           like <|im_start|>user\n...<|im_end|> per the model's training format.
        2. Fallback (MI-5): For tokenizers without chat_template, concatenates
           messages as "{role}: {content}\n" and appends "assistant: " if
           add_generation_prompt is True.

    Example:
        POST /v1/chat/completions
        {
          "model": "qwen",
          "messages": [
            {"role": "system", "content": "You are a poet."},
            {"role": "user", "content": "Write a haiku about coding."}
          ],
          "max_tokens": 64,
          "temperature": 0.8,
          "stream": true
        }
    """
    model: Optional[str] = None
    """Model identifier. Currently unused (single-model deployment)."""

    messages: list[ChatMessage]
    """Chat conversation history. Must contain at least one message."""

    max_tokens: Optional[int] = 16
    """Maximum tokens to generate. Default 16 matches OpenAI API."""

    temperature: Optional[float] = 1.0
    """Sampling temperature. Clamped to >= 1e-6 (CR-1)."""

    stream: Optional[bool] = False
    """Whether to stream results as SSE."""

    stop: Optional[Union[str, list[str]]] = None
    """Stop sequences. Accepted but not enforced in Phase 4."""

    ignore_eos: bool = False
    """Whether to ignore the EOS token."""

    user: Optional[str] = None
    """User identifier for abuse monitoring. Accepted but unused."""

    add_generation_prompt: bool = True
    """
    Whether to append a generation prompt to the chat template output.

    When True (default): The chat template includes the assistant's turn marker,
    signaling the model to start generating. Example output:
      <|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n

    When False: Only the conversation history is encoded, no turn marker added.
    Useful when you want raw conversation encoding without triggering generation.
    """

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v):
        """
        Validate that messages list is non-empty.

        An empty messages list would produce an empty prompt, which is an invalid
        state for the engine (no input tokens to process).

        Args:
            v: The messages list.

        Returns:
            The original list if non-empty.

        Raises:
            ValueError: If messages list is empty.
        """
        if len(v) == 0:
            raise ValueError(
                "messages must not be empty. "
                "At least one message (e.g., a user message) is required."
            )
        return v


class ChatCompletionResponseChoice(BaseModel):
    """
    A single choice in a chat completion response (non-streaming).

    Format:
      {
        "index": 0,
        "message": {"role": "assistant", "content": "Hi!"},
        "finish_reason": "stop"
      }
    """
    index: int = 0
    """Choice index. Always 0 in single-completion mode."""

    message: ChatMessage = ChatMessage(role="assistant", content="")
    """The generated assistant message."""

    finish_reason: Optional[str] = None
    """Reason for termination: "stop", "length", or None (if streaming)."""


class ChatCompletionResponse(OpenAIBaseModel):
    """
    Full response body for POST /v1/chat/completions (non-streaming).

    Format:
      {
        "id": "chatcmpl-a1b2c3d4",
        "object": "chat.completion",
        "created": 1712345678,
        "model": "Qwen2-0.5B-Instruct",
        "choices": [...],
        "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
        "metrics": {"ttft": 0.05, "tpot": 0.01, "total_time": 1.23}
      }
    """
    id: str
    """Request ID, e.g., "chatcmpl-a1b2c3d4"."""

    object: str = "chat.completion"
    """Always "chat.completion"."""

    created: int
    """Unix timestamp of response creation."""

    model: str
    """Model name."""

    choices: list[ChatCompletionResponseChoice]
    """Array of completion choices."""

    usage: UsageInfo
    """Token usage statistics."""

    metrics: Optional[MetricsInfo] = None
    """Per-request performance timing metrics (baby-vllm extension)."""


class DeltaMessage(BaseModel):
    """
    A delta (partial update) message for streaming chat completions.

    Unlike ChatMessage which contains the full message, DeltaMessage contains
    only the fields that changed since the previous chunk.

    Streaming pattern:
      Chunk 1: DeltaMessage(role="assistant", content="Hello")
               → Signals: "this is the assistant speaking, first word is 'Hello'"
      Chunk 2: DeltaMessage(content=" world")
               → Signals: "append ' world' to the content"
      Chunk 3: DeltaMessage(content="!")
               → Signals: "append '!' to the content"
      Final: DeltaMessage(content="") + finish_reason="stop"
             → Signals: "generation complete"

    The role is only set in the first chunk. Subsequent chunks only update content.
    """
    role: Optional[str] = None
    """Assistant role. Only set in the FIRST streaming chunk. None thereafter."""

    content: Optional[str] = None
    """Text delta to append. Empty string in the final chunk."""


class ChatCompletionResponseStreamChoice(BaseModel):
    """
    A single choice within a streaming chat completion SSE chunk.

    Format:
      {
        "index": 0,
        "delta": {"role": "assistant", "content": "Hello"},
        "finish_reason": null
      }
    """
    index: int = 0
    """Choice index. Always 0."""

    delta: DeltaMessage = DeltaMessage()
    """The delta update for this chunk."""

    finish_reason: Optional[str] = None
    """Non-null only in the final chunk."""


class ChatCompletionStreamResponse(OpenAIBaseModel):
    """
    A single SSE chunk in a streaming /v1/chat/completions response.

    Each chunk:
      data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"...","choices":[...]}

    Usage and metrics are only included in the FINAL chunk (when finish_reason is set).
    """
    id: str
    """Same request ID across all chunks."""

    object: str = "chat.completion.chunk"
    """Always "chat.completion.chunk"."""

    created: int
    """Same timestamp across all chunks."""

    model: str
    """Same model name across all chunks."""

    choices: list[ChatCompletionResponseStreamChoice]
    """Streaming choices for this chunk."""

    usage: Optional[UsageInfo] = None
    """
    Token usage statistics. Only included in the FINAL chunk.
    Set to None for intermediate chunks.
    """

    metrics: Optional[MetricsInfo] = None
    """
    Per-request performance timing metrics. Only included in the FINAL chunk.
    Set to None for intermediate chunks.
    """


# ===========================================================================
# Pydantic Models — /v1/models
# ===========================================================================


class ModelCard(BaseModel):
    """
    A single model entry in the /v1/models listing.

    Format:
      {
        "id": "Qwen2-0.5B-Instruct",
        "object": "model",
        "created": 1712345678,
        "owned_by": "baby-vllm"
      }
    """
    id: str
    """Model identifier. Usually the directory/hub name."""

    object: str = "model"
    """Always "model"."""

    created: int
    """Unix timestamp. Uses current time (not model creation time) for simplicity."""

    owned_by: str = "baby-vllm"
    """Organization identifier. Hardcoded to "baby-vllm"."""


class ModelList(BaseModel):
    """
    Response body for GET /v1/models.

    Format:
      {
        "object": "list",
        "data": [ModelCard, ...]
      }
    """
    object: str = "list"
    """Always "list"."""

    data: list[ModelCard]
    """List of available model cards. In baby-vllm, always a single-element list."""


# ===========================================================================
# FastAPI Application
# ===========================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup/shutdown hooks.

    Startup:
      Nothing to do — the engine is created by CLI launcher before uvicorn.run(),
      and injected via api_server._engine = engine. This ensures the engine is
      fully initialized (model loaded, workers started) before the server accepts
      any requests.

    Shutdown:
      Calls _engine.stop() to gracefully:
        1. Stop the background engine loop (set stop event, cancel task)
        2. Clean up in-flight request streams
        3. LLMEngine worker processes are cleaned up separately by atexit handler

    Why not create the engine here?
      The engine creation is expensive (model loading, GPU memory allocation).
      Doing it here would mean the server starts listening before the model is
      ready, which is confusing for health checks. The CLI launcher pattern
      ensures the server only starts after the model is fully loaded.

    Usage:
      app = FastAPI(lifespan=lifespan)
    """
    # Startup: yield allows the app to start serving
    yield
    # Shutdown: cleanup after the server stops
    if _engine is not None:
        await _engine.stop()


# Create the FastAPI application instance
app = FastAPI(
    title="baby-vllm API Server",
    description="OpenAI-compatible REST API for the baby-vllm inference engine",
    version="0.1.0",
    lifespan=lifespan,
)


# ===========================================================================
# CORS Middleware (MI-3)
# ===========================================================================
#
# Why allow all origins?
#   baby-vllm is primarily used for local development and research.
#   Browser-based API consumers (e.g., web playgrounds, Jupyter notebooks)
#   need CORS to make cross-origin requests. For production, users should
#   restrict this to specific origins.
# ===========================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# Error Handlers (MI-4) — OpenAI-compatible error format
# ===========================================================================


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Render HTTPException as OpenAI-compatible error JSON.

    OpenAI error format:
      {
        "error": {
          "message": "The model does not exist",
          "type": "invalid_request_error",
          "param": "model",
          "code": 404
        }
      }

    Why match this format?
      OpenAI SDK clients (openai Python package, etc.) parse error responses
      expecting this structure. Using a different format would break error
      handling in client code.

    The 'param' and additional fields are omitted for simplicity but can
    be added in future phases.
    """
    return Response(
        status_code=exc.status_code,
        content=json.dumps({
            "error": {
                "message": exc.detail,
                "type": "invalid_request_error",
                "code": exc.status_code,
            }
        }),
        media_type="application/json",
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all handler for unexpected server errors.

    Logs the traceback to stderr and returns a 500 error in OpenAI format.
    This prevents stack traces from leaking to clients while still providing
    meaningful error messages.
    """
    import traceback
    traceback.print_exc()
    return Response(
        status_code=500,
        content=json.dumps({
            "error": {
                "message": f"Internal server error: {str(exc)}",
                "type": "internal_error",
                "code": 500,
            }
        }),
        media_type="application/json",
    )


# ===========================================================================
# Endpoints
# ===========================================================================


@app.get("/health")
async def health():
    """
    Health check endpoint (GET /health).

    Returns:
        200 OK if engine is initialized and ready.
        503 Service Unavailable if engine is not yet initialized.

    Purpose:
      Used by load balancers, Docker health checks, and monitoring tools
      to verify the server is running AND the engine is ready.
      Checks _engine state to distinguish "server up" from "model loaded".

    Example:
        $ curl http://localhost:8000/health
        (HTTP 200, empty body)
    """
    # Review fix: Check engine is initialized, not just FastAPI accepting connections.
    # Without this, a health check could pass while the model is still loading.
    if _engine is None:
        return Response(status_code=503)
    return Response(status_code=200)


@app.get("/v1/models")
async def show_available_models():
    """
    List available models (GET /v1/models).

    Returns a ModelList containing the currently loaded model. In baby-vllm,
    only one model is loaded at a time (single-model deployment).

    Returns:
        ModelList with a single ModelCard.

    Raises:
        HTTPException(503): If engine is not initialized.

    Example:
        GET /v1/models
        Response:
        {
          "object": "list",
          "data": [
            {
              "id": "Qwen2-0.5B-Instruct",
              "object": "model",
              "created": 1712345678,
              "owned_by": "baby-vllm"
            }
          ]
        }
    """
    if _engine is None:
        raise HTTPException(
            status_code=503,
            detail="Engine not initialized. Start the server with babyvllm-server --model <path>",
        )

    model_id = _get_model_name()
    card = ModelCard(
        id=model_id,
        created=int(time.time()),
        owned_by="baby-vllm",
    )
    return ModelList(data=[card])


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    """
    Create a text completion (POST /v1/completions).

    This is the OpenAI /v1/completions endpoint. Supports both:
      - Non-streaming (stream=False): Returns a single CompletionResponse JSON
      - Streaming (stream=True): Returns an SSE (text/event-stream) response

    Processing Pipeline:
      1. Validate request via Pydantic (already done by FastAPI)
      2. Normalize prompt → pass through (validation already done)
      3. Clamp temperature → prevent SamplingParams assertion error (CR-1)
      4. Build SamplingParams from request fields
      5. Generate string request_id (CR-3)
      6. Choose streaming or non-streaming code path
      7. For streaming: return StreamingResponse wrapping SSE async generator
      8. For non-streaming: collect all outputs, return final JSON

    Args:
        request: Validated CompletionRequest from request body.
        raw_request: Raw FastAPI Request object. Accepted for future use
                     (e.g., client disconnect detection). Currently unused.

    Returns:
        CompletionResponse JSON (non-streaming) or
        StreamingResponse with text/event-stream (streaming).

    Raises:
        HTTPException(503): If engine is not initialized.

    Example (non-streaming):
        POST /v1/completions
        {"model": "qwen", "prompt": "Hello", "max_tokens": 32}

    Example (streaming):
        POST /v1/completions
        {"model": "qwen", "prompt": "Hello", "max_tokens": 32, "stream": true}
    """
    if _engine is None:
        raise HTTPException(
            status_code=503,
            detail="Engine not initialized. Start the server with babyvllm-server --model <path>",
        )

    # (1) Normalize prompt — currently pass-through, Pydantic already validated format
    prompt = _normalize_prompt(request.prompt)

    # (2) Clamp temperature — CR-1: avoid SamplingParams assertion for temperature=0
    temperature = _clamp_temperature(request.temperature)

    # (3) Build SamplingParams from request
    #     - temperature: clamped to >= 1e-6
    #     - max_tokens: from request, default 16 matching OpenAI API
    #     - ignore_eos: from request, default False
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=request.max_tokens if request.max_tokens is not None else 16,
        ignore_eos=request.ignore_eos,
    )

    # (4) Get model name for response headers
    model_name = _get_model_name()

    # (5) Generate API-layer string request_id (CR-3)
    #     NOT using engine's internal int ID — we generate our own UUID-based string
    request_id = _random_id("cmpl")

    # (6) Route to streaming or non-streaming handler
    if request.stream:
        # Streaming: SSE response via async generator
        # Headers:
        #   Cache-Control: no-cache — prevent proxies from buffering the event stream
        #   Connection: keep-alive — maintain persistent connection for SSE
        return StreamingResponse(
            _stream_completion(
                prompt, sampling_params, model_name, request_id,
                request.echo, raw_request,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    else:
        # Non-streaming: collect all outputs, return single JSON response
        return await _non_stream_completion(
            prompt, sampling_params, model_name, request_id, request.echo,
        )


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    """
    Create a chat completion (POST /v1/chat/completions).

    This is the OpenAI /v1/chat/completions endpoint. Supports both streaming
    and non-streaming modes.

    Chat Message → Token IDs Conversion:
      Messages are converted to token IDs using the model's chat template.
      Two strategies (in priority order):
        1. Chat Template: Uses tokenizer.apply_chat_template() with the model's
           built-in Jinja2 template. Adds special tokens per model format.
           Example: <|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n
        2. Fallback (MI-5): Simple format for tokenizers without chat_template:
           "system: You are helpful.\nuser: Hi\nassistant: "

    Processing Pipeline:
      1. Convert chat messages → token IDs (via chat template or fallback)
      2. Clamp temperature (CR-1)
      3. Build SamplingParams
      4. Generate string request_id (CR-3)
      5. Route to streaming or non-streaming handler

    Args:
        request: Validated ChatCompletionRequest from request body.
        raw_request: Raw FastAPI Request object (for future disconnect detection).

    Returns:
        ChatCompletionResponse JSON (non-streaming) or
        StreamingResponse with text/event-stream (streaming).

    Raises:
        HTTPException(503): If engine is not initialized.

    Example:
        POST /v1/chat/completions
        {
          "model": "qwen",
          "messages": [
            {"role": "user", "content": "Write a haiku about coding."}
          ],
          "max_tokens": 64
        }
    """
    if _engine is None:
        raise HTTPException(
            status_code=503,
            detail="Engine not initialized. Start the server with babyvllm-server --model <path>",
        )

    # (1) Convert chat messages to prompt token IDs
    #     First convert Pydantic ChatMessage → plain dict for tokenizer
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # (1a) Preferred path: Use tokenizer's built-in chat_template
    #      tokenize=True → returns list[int] (token IDs), not str
    #      add_generation_prompt → appends the assistant turn marker
    #
    # (1b) Fallback (MI-5): Tokenizers without chat_template (e.g., older models)
    #      Concatenate messages as "{role}: {content}" lines
    #      Append "assistant: " if add_generation_prompt=True
    if hasattr(_engine.tokenizer, "chat_template") and _engine.tokenizer.chat_template:
        prompt_token_ids = _engine.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=request.add_generation_prompt,
        )
        # transformers >= 4.35 returns BatchEncoding (dict-like) instead of list[int].
        # Extract input_ids and convert to plain list if needed.
        if hasattr(prompt_token_ids, "input_ids"):
            prompt_token_ids = prompt_token_ids.input_ids
            if hasattr(prompt_token_ids, "tolist"):  # torch.Tensor / numpy array
                prompt_token_ids = prompt_token_ids.tolist()
    else:
        # MI-5: Fallback for tokenizers without chat_template
        fallback_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )
        if request.add_generation_prompt:
            fallback_text += "\nassistant: "
        prompt_token_ids = _engine.tokenizer.encode(fallback_text)

    # (2) Clamp temperature (CR-1)
    temperature = _clamp_temperature(request.temperature)

    # (3) Build SamplingParams
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=request.max_tokens if request.max_tokens is not None else 16,
        ignore_eos=request.ignore_eos,
    )

    # (4) Model name and request ID
    model_name = _get_model_name()
    request_id = _random_id("chatcmpl")

    # (5) Route to streaming or non-streaming
    if request.stream:
        return StreamingResponse(
            _stream_chat_completion(
                prompt_token_ids, sampling_params, model_name, request_id,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    else:
        return await _non_stream_chat_completion(
            prompt_token_ids, sampling_params, model_name, request_id,
        )


# ===========================================================================
# Non-Streaming Helpers
# ===========================================================================
#
# These functions collect ALL RequestOutput from the async generator and
# return a single Pydantic model for JSON serialization.
#
# Why collect all outputs instead of just the first/last?
#   In the current engine implementation, engine.step() only produces output
#   when a sequence is FINISHED. So each request typically has exactly 1 output.
#   However, future incremental streaming optimizations may produce multiple
#   outputs per request (e.g., per-token decoding). Collecting all and using
#   the last ensures correctness regardless of engine behavior.
# ===========================================================================


async def _non_stream_completion(
    prompt: Union[str, list[int]],
    sampling_params: SamplingParams,
    model_name: str,
    request_id: str,
    echo: bool,
) -> CompletionResponse:
    """
    Non-streaming /v1/completions handler.

    Collects all outputs from the engine, takes the final output, and builds
    a CompletionResponse.

    Args:
        prompt: Raw prompt (str or list[int]).
        sampling_params: Sampling parameters.
        model_name: Model name for the response "model" field.
        request_id: API-layer string request ID (CR-3).
        echo: Whether to prepend the prompt to the generated text.

    Returns:
        CompletionResponse with the full generated text, usage stats, and metadata.

    Raises:
        HTTPException(500): If no output was generated (engine returned empty).
    """
    final_output: Optional[RequestOutput] = None
    engine_request_id: Optional[int] = None  # Capture engine's internal int ID for abort

    # Consume all outputs from the async generator.
    # In current engine, only 1 output per request (when finished).
    # The for-loop pattern future-proofs for incremental decoding.
    try:
        async for output in _engine.generate(prompt, sampling_params):
            if engine_request_id is None:
                engine_request_id = output.request_id
            final_output = output
    except asyncio.CancelledError:
        # Client disconnected — ensure engine-side cleanup.
        # Although AsyncStream.aclose() also triggers abort via the cancel hook,
        # explicitly calling abort() provides defense-in-depth and cleans up
        # AsyncLLMEngine internal mappings (_request_to_seq, _seq_to_request, _prompt_map).
        #
        # Use engine_request_id if captured from output, otherwise fall back to the
        # engine's _current_request_id (set in generate() before first yield).
        # This handles client disconnect before any RequestOutput is generated.
        abort_id = engine_request_id if engine_request_id is not None else _engine._current_request_id
        if abort_id is not None:
            _engine.abort(abort_id)
        raise  # Re-raise so FastAPI knows the request was cancelled

    if final_output is None:
        raise HTTPException(status_code=500, detail="No output generated")

    # Build text: with or without echo
    # Echo prepends the decoded prompt to the generated text
    # Example:
    #   Prompt: "Hello, my name is" (tokenized to [15496, 11, ...])
    #   Generated: " Bob"
    #   echo=False → " Bob"
    #   echo=True  → "Hello, my name is Bob"
    text = final_output.text
    if echo:
        prompt_text = _engine.tokenizer.decode(final_output.prompt_token_ids)
        text = prompt_text + text

    # Build the single choice
    choice = CompletionResponseChoice(
        index=0,
        text=text,
        finish_reason="stop" if final_output.finished else None,
    )

    # CR-4: Compute usage info from token ID arrays
    # prompt_tokens = number of input tokens
    # completion_tokens = number of generated tokens (new tokens only)
    usage = UsageInfo(
        prompt_tokens=len(final_output.prompt_token_ids),
        completion_tokens=len(final_output.token_ids),
        total_tokens=len(final_output.prompt_token_ids) + len(final_output.token_ids),
    )

    # Extract per-request timing metrics from RequestOutput (if available).
    # These are populated by AsyncLLMEngine for completed requests.
    metrics = None
    if final_output.ttft is not None or final_output.tpot is not None or final_output.total_time is not None:
        metrics = MetricsInfo(
            ttft=final_output.ttft,
            tpot=final_output.tpot,
            total_time=final_output.total_time,
        )

    # MI-1: Return Pydantic model directly — FastAPI handles JSON serialization
    return CompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=model_name,
        choices=[choice],
        usage=usage,
        metrics=metrics,
    )


async def _non_stream_chat_completion(
    prompt_token_ids: list[int],
    sampling_params: SamplingParams,
    model_name: str,
    request_id: str,
) -> ChatCompletionResponse:
    """
    Non-streaming /v1/chat/completions handler.

    Similar to _non_stream_completion but wraps the generated text in a
    ChatMessage with role="assistant".

    Args:
        prompt_token_ids: Pre-tokenized prompt (from chat template).
        sampling_params: Sampling parameters.
        model_name: Model name for response.
        request_id: API-layer string request ID.

    Returns:
        ChatCompletionResponse with assistant message, usage stats, and metadata.

    Raises:
        HTTPException(500): If no output was generated.
    """
    final_output: Optional[RequestOutput] = None
    engine_request_id: Optional[int] = None  # Capture engine's internal int ID for abort

    try:
        async for output in _engine.generate(prompt_token_ids, sampling_params):
            if engine_request_id is None:
                engine_request_id = output.request_id
            final_output = output
    except asyncio.CancelledError:
        # Client disconnected — ensure engine-side cleanup.
        # Although AsyncStream.aclose() also triggers abort via the cancel hook,
        # explicitly calling abort() provides defense-in-depth.
        #
        # Use engine_request_id if captured from output, otherwise fall back to the
        # engine's _current_request_id (set in generate() before first yield).
        # This handles client disconnect before any RequestOutput is generated.
        abort_id = engine_request_id if engine_request_id is not None else _engine._current_request_id
        if abort_id is not None:
            _engine.abort(abort_id)
        raise  # Re-raise so FastAPI knows the request was cancelled

    if final_output is None:
        raise HTTPException(status_code=500, detail="No output generated")

    # Wrap generated text in ChatMessage with role="assistant"
    choice = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=final_output.text),
        finish_reason="stop" if final_output.finished else None,
    )

    # CR-4: Compute usage info
    usage = UsageInfo(
        prompt_tokens=len(final_output.prompt_token_ids),
        completion_tokens=len(final_output.token_ids),
        total_tokens=len(final_output.prompt_token_ids) + len(final_output.token_ids),
    )

    # Extract per-request timing metrics from RequestOutput (if available).
    metrics = None
    if final_output.ttft is not None or final_output.tpot is not None or final_output.total_time is not None:
        metrics = MetricsInfo(
            ttft=final_output.ttft,
            tpot=final_output.tpot,
            total_time=final_output.total_time,
        )

    # MI-1: Return Pydantic model directly
    return ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=model_name,
        choices=[choice],
        usage=usage,
        metrics=metrics,
    )


# ===========================================================================
# Streaming Helpers (SSE Generators)
# ===========================================================================
#
# These are async generators yielding SSE-formatted strings.
# SSE format: "data: {json}\n\n"
# The "data: [DONE]\n\n" sentinel signals the end of the stream.
#
# Cancellation Handling:
#   When the client disconnects, asyncio.CancelledError is raised inside the
#   async for loop. We catch it silently to avoid stack trace noise.
#   The underlying AsyncStream.aclose() automatically handles cleanup
#   (aborts the request in the engine's scheduler).
# ===========================================================================


async def _stream_completion(
    prompt: Union[str, list[int]],
    sampling_params: SamplingParams,
    model_name: str,
    request_id: str,
    echo: bool,
    raw_request: Request,
) -> AsyncGenerator[str, None]:
    """
    SSE streaming generator for /v1/completions.

    Yields SSE-formatted strings (one per engine output), then [DONE] sentinel.

    Streaming Pattern:
      data: {"id":"cmpl-...","object":"text_completion","choices":[...]}\n\n
      data: {"id":"cmpl-...","object":"text_completion","choices":[...]}\n\n
      data: [DONE]\n\n

    Echo in streaming:
      The echo prefix is prepended only to the FIRST chunk, using a simple
      flag (_echo_prefix_sent) attached to the output object. This avoids
      repeating the prompt text in every chunk.

    Args:
        prompt: Raw prompt (str or list[int]).
        sampling_params: Sampling parameters.
        model_name: Model name for each chunk.
        request_id: API-layer string request ID.
        echo: Whether to include prompt text in the output.
        raw_request: FastAPI Request object (for future disconnect detection).

    Yields:
        str: SSE-formatted data lines.
    """
    created = int(time.time())
    engine_request_id: Optional[int] = None  # Capture engine's internal int ID for abort
    try:
        async for output in _engine.generate(prompt, sampling_params):
            # Capture engine request_id on first output for abort() calls.
            # output.request_id is the engine's internal integer ID (from
            # _request_counter), NOT the API-layer string request_id (from _random_id).
            # We need this to call _engine.abort() in the CancelledError handler.
            if engine_request_id is None:
                engine_request_id = output.request_id

            text = output.text

            # Echo: prepend prompt text on first chunk only
            # Uses a monkey-patched attribute on the output object as a flag.
            # This is a pragmatic solution — avoids maintaining external state
            # while keeping the async generator self-contained.
            if echo and not getattr(output, "_echo_prefix_sent", False):
                prompt_text = _engine.tokenizer.decode(output.prompt_token_ids)
                text = prompt_text + text
                # Mark that echo prefix has been sent for this output stream
                output._echo_prefix_sent = True  # type: ignore[attr-defined]

            choice = CompletionResponseChoice(
                index=0,
                text=text,
                finish_reason="stop" if output.finished else None,
            )

            # Include usage and metrics only in the final chunk.
            usage = None
            metrics = None
            if output.finished:
                usage = UsageInfo(
                    prompt_tokens=len(output.prompt_token_ids),
                    completion_tokens=len(output.token_ids),
                    total_tokens=len(output.prompt_token_ids) + len(output.token_ids),
                )
                if output.ttft is not None or output.tpot is not None or output.total_time is not None:
                    metrics = MetricsInfo(
                        ttft=output.ttft,
                        tpot=output.tpot,
                        total_time=output.total_time,
                    )

            chunk = CompletionStreamResponse(
                id=request_id,
                created=created,
                model=model_name,
                choices=[choice],
                usage=usage,
                metrics=metrics,
            )

            # SSE format: "data: <json>\n\n"
            # exclude_none=True omits usage=null and metrics=null from
            # intermediate chunks, matching the pattern used in chat streaming.
            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

        # Signal end of stream
        yield "data: [DONE]\n\n"

    except asyncio.CancelledError:
        # Client disconnected — ensure engine-side cleanup.
        # Although AsyncStream.aclose() also triggers abort via the cancel hook,
        # explicitly calling abort() here provides defense-in-depth:
        #   - Guarantees cleanup even if aclose() is not called (edge case)
        #   - Makes the cleanup path visible in the code
        #   - Cleans up AsyncLLMEngine internal mappings that the
        #     tracker alone cannot touch (_request_to_seq, _seq_to_request, _prompt_map)
        #
        # Use engine_request_id if captured from output, otherwise fall back to the
        # engine's _current_request_id (set in generate() before first yield).
        # This handles client disconnect before any RequestOutput is generated.
        abort_id = engine_request_id if engine_request_id is not None else _engine._current_request_id
        if abort_id is not None:
            _engine.abort(abort_id)
        # Do NOT re-raise. Starlette's StreamingResponse internally catches
        # CancelledError when client disconnects. Re-raising would cause
        # unexpected behavior in the HTTP layer.
        pass


async def _stream_chat_completion(
    prompt_token_ids: list[int],
    sampling_params: SamplingParams,
    model_name: str,
    request_id: str,
) -> AsyncGenerator[str, None]:
    """
    SSE streaming generator for /v1/chat/completions.

    Yields SSE-formatted strings for each engine output chunk.

    Streaming Pattern for Chat:
      Chunk 1 (has role):
        data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}\n\n
      Chunk 2+ (no role):
        data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"delta":{"content":" world"},"finish_reason":null}]}\n\n
      Final chunk:
        data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"delta":{"content":""},"finish_reason":"stop"}],"usage":{...}}\n\n
        data: [DONE]\n\n

    Design Decision — First chunk includes role:
      The role ("assistant") is only sent in the first chunk. This matches
      OpenAI's streaming behavior:
      - The first chunk tells the client "this is the assistant speaking"
      - Subsequent chunks only deliver content deltas
      - The final chunk includes finish_reason and usage

    Args:
        prompt_token_ids: Pre-tokenized prompt (from chat template).
        sampling_params: Sampling parameters.
        model_name: Model name for each chunk.
        request_id: API-layer string request ID.

    Yields:
        str: SSE-formatted data lines.
    """
    created = int(time.time())
    is_first: bool = True  # Track whether this is the first chunk for role inclusion
    engine_request_id: Optional[int] = None  # Capture engine's internal int ID for abort
    try:
        async for output in _engine.generate(prompt_token_ids, sampling_params):
            # Capture engine request_id on first output for abort() calls.
            # output.request_id is the engine's internal integer ID (from
            # _request_counter), NOT the API-layer string request_id.
            if engine_request_id is None:
                engine_request_id = output.request_id

            delta = DeltaMessage()

            # First chunk: include role="assistant"
            # This signals to the client that the assistant is about to speak.
            if is_first:
                delta.role = "assistant"
                is_first = False

            # Always include the content delta
            # In current engine: this is the full generated text (only one output per request)
            # In future: this will be per-token deltas
            delta.content = output.text

            choice = ChatCompletionResponseStreamChoice(
                index=0,
                delta=delta,
                finish_reason="stop" if output.finished else None,
            )

            # Usage and metrics are only included in the final chunk
            usage = None
            metrics = None
            if output.finished:
                usage = UsageInfo(
                    prompt_tokens=len(output.prompt_token_ids),
                    completion_tokens=len(output.token_ids),
                    total_tokens=len(output.prompt_token_ids) + len(output.token_ids),
                )
                if output.ttft is not None or output.tpot is not None or output.total_time is not None:
                    metrics = MetricsInfo(
                        ttft=output.ttft,
                        tpot=output.tpot,
                        total_time=output.total_time,
                    )

            chunk = ChatCompletionStreamResponse(
                id=request_id,
                created=created,
                model=model_name,
                choices=[choice],
                usage=usage,
                metrics=metrics,
            )

            # exclude_none=True (Review fix): OpenAI omits null fields from
            # chat streaming chunks. Without this, intermediate chunks would
            # serialize {"usage": null, "delta": {"role": null, "content": "..."}}.
            # We need exclude_none to omit role=null in non-first chunks
            # and usage=null in non-final chunks.
            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

        # Signal end of stream
        yield "data: [DONE]\n\n"

    except asyncio.CancelledError:
        # Client disconnected — ensure engine-side cleanup (defense-in-depth).
        # Although AsyncStream.aclose() also triggers abort via the cancel hook,
        # explicitly calling abort() here provides defense-in-depth:
        #   - Guarantees cleanup even if aclose() is not called (edge case)
        #   - Makes the cleanup path visible in the code
        #   - Cleans up AsyncLLMEngine internal mappings
        #
        # Use engine_request_id if captured from output, otherwise fall back to the
        # engine's _current_request_id (set in generate() before first yield).
        # This handles client disconnect before any RequestOutput is generated.
        abort_id = engine_request_id if engine_request_id is not None else _engine._current_request_id
        if abort_id is not None:
            _engine.abort(abort_id)
        pass
