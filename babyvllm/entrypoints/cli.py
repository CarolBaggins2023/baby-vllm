#!/usr/bin/env python3
"""
baby-vllm CLI — Launcher for the OpenAI-compatible API Server
==============================================================

Core Responsibilities:
  Parse command-line arguments to configure the baby-vllm inference engine,
  create an AsyncLLMEngine instance, inject it into the API server module,
  and start the uvicorn HTTP server.

=============================================================
Startup Flow
=============================================================

  1. Argument Parsing (argparse)
     └── Parse model path, server settings, engine config, logging level

  2. Engine Creation
     └── AsyncLLMEngine(model=args.model, **engine_kwargs)
         ├── Load model weights (GPU memory allocation)
         ├── Start multiprocessing worker processes
         ├── Create tokenizer (HuggingFace AutoTokenizer)
         └── Create scheduler + KV cache block manager

  3. Engine Injection
     └── api_server._engine = engine
         (Module-level singleton pattern — all endpoints share this engine)

  4. Server Start
     └── uvicorn.run(api_server.app, host=..., port=..., log_level=...)
         (Blocking call — runs until interrupted by SIGTERM/SIGINT)

  5. Shutdown
     └── FastAPI lifespan → api_server._engine.stop()
         ├── Stop background engine loop
         └── atexit → LLMEngine.exit() → cleanup worker processes

Usage:
    python -m babyvllm.entrypoints.cli --model /path/to/model --port 8000

    # Or via the console_scripts entry point (after pip install):
    babyvllm-server --model /path/to/model --port 8000
"""

import argparse


def create_parser() -> argparse.ArgumentParser:
    """
    Create and return the argument parser for the baby-vllm server CLI.

    Returns:
        argparse.ArgumentParser: Configured argument parser (not yet parsed).

    Example:
        >>> parser = create_parser()
        >>> args = parser.parse_args(["--model", "/path/to/model"])
    """
    parser = argparse.ArgumentParser(
        prog="baby-vllm-server",
        description="baby-vllm OpenAI-compatible API server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with default settings
  babyvllm-server --model /data/models/Qwen2-0.5B-Instruct

  # Custom port and GPU settings
  babyvllm-server --model /data/models/Qwen2-0.5B-Instruct --port 8080 --tensor-parallel-size 2

  # Disable CUDA graphs for debugging
  babyvllm-server --model /data/models/Qwen2-0.5B-Instruct --enforce-eager

  # Verbose logging for troubleshooting
  babyvllm-server --model /data/models/Qwen2-0.5B-Instruct --log-level debug
        """,
    )

    # ---- Required Arguments ----

    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to the model directory (local filesystem path). "
             "Example: /data/models/Qwen2-0.5B-Instruct",
    )

    # ---- Server Configuration ----

    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Server host address to bind to. "
             "Default: 127.0.0.1 (localhost only). "
             "Use 0.0.0.0 to accept connections from all network interfaces.",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Server port to listen on. Default: 8000.",
    )

    # ---- Engine / Model Configuration ----
    # These map to Config dataclass fields (see babyvllm/config.py).
    # Names use hyphen-separated CLI convention (e.g., --max-model-len),
    # which argparse maps to underscore-separated Python attributes (max_model_len).

    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1,
        help="Number of tensor parallel replicas. "
             "Splits model weights across multiple GPUs. "
             "Default: 1 (single GPU). Range: 1-8.",
    )
    parser.add_argument(
        "--max-num-batched-tokens", type=int, default=16384,
        help="Maximum total tokens per batch per inference step. "
             "Limits VRAM usage during prefill. "
             "Default: 16384 (matching Config default).",
    )
    parser.add_argument(
        "--max-num-sequences", type=int, default=512,
        help="Maximum number of concurrent sequences in the scheduler. "
             "Limits the number of in-flight requests. "
             "Default: 512 (matching Config default).",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=None,
        help="Override the maximum model context length. "
             "If not set, uses the model's default max_position_embeddings from HF config, "
             "capped at Config.max_model_length (4096 by default). "
             "Must be <= max-num-batched-tokens. "
             "Maps to Config.max_model_length field.",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.9,
        help="Fraction of GPU memory to allocate for KV cache. "
             "Default: 0.9 (use 90%% of available GPU memory). "
             "Lower values reserve memory for other processes. "
             "Maps to Config.gpu_memory_utilization.",
    )
    parser.add_argument(
        "--enforce-eager", action="store_true", default=False,
        help="Disable CUDA graph optimization. "
             "CUDA graphs improve throughput by reducing kernel launch overhead, "
             "but can cause issues with dynamic shapes or debugging. "
             "Enable this flag for debugging or on GPUs without CUDA graph support. "
             "Maps to Config.enforce_eager.",
    )

    # ---- KV Cache Configuration (CR-5) ----

    parser.add_argument(
        "--kvcache-block-size", type=int, default=256,
        help="Size of each KV cache block in tokens. "
             "Must be a multiple of 256. "
             "Larger blocks → fewer blocks to manage, but more potential memory waste. "
             "Default: 256 (matching Config default). "
             "Maps to Config.kvcache_block_size.",
    )
    parser.add_argument(
        "--num-kvcache-blocks", type=int, default=-1,
        help="Number of KV cache blocks to allocate. "
             "Default: -1 (auto-compute based on GPU memory utilization). "
             "Override only for debugging or precise memory control. "
             "Maps to Config.num_kvcache_blocks.",
    )

    # ---- Logging ----

    parser.add_argument(
        "--log-level", type=str, default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level for uvicorn server output. "
             "Default: info. Use debug for verbose request/response logging.",
    )

    return parser


def build_engine_kwargs(args: argparse.Namespace) -> dict:
    """
    Build the engine keyword arguments dict from parsed CLI arguments.

    Maps CLI argument names to Config field names.  Note naming differences:
      CLI: --max-model-len       → Config: max_model_length
      CLI: --kvcache-block-size  → Config: kvcache_block_size
      CLI: --num-kvcache-blocks  → Config: num_kvcache_blocks

    All kwargs are passed to AsyncLLMEngine.__init__(), which forwards them to
    LLMEngine.__init__() → Config.__init__().

    Args:
        args: Parsed argparse Namespace from create_parser().

    Returns:
        dict: Keyword arguments for AsyncLLMEngine.__init__().
    """
    engine_kwargs = {
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_sequences": args.max_num_sequences,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "enforce_eager": args.enforce_eager,
        "kvcache_block_size": args.kvcache_block_size,
        "num_kvcache_blocks": args.num_kvcache_blocks,
        "host": args.host,
        "port": args.port,
    }

    # max_model_len is optional — only set if user explicitly provided it.
    # If None, Config uses its own default (min of max_model_length and
    # the model's max_position_embeddings).
    if args.max_model_len is not None:
        engine_kwargs["max_model_length"] = args.max_model_len

    return engine_kwargs


def main():
    """
    Entry point for the baby-vllm-server command.

    Parses CLI arguments, creates the engine, and starts the server.

    Exit Codes:
      0 — Normal shutdown (SIGTERM/SIGINT)
      1 — Error during engine creation (model not found, GPU OOM, etc.)

    Example:
        $ babyvllm-server --model /data/models/Qwen2-0.5B-Instruct --port 8000
        Loading model from /data/models/Qwen2-0.5B-Instruct...
        Model loaded successfully.
        Starting server on 127.0.0.1:8000...
    """
    # =====================================================================
    # Phase 1: Argument Parsing
    # =====================================================================

    parser = create_parser()
    args = parser.parse_args()

    # =====================================================================
    # Phase 2: Build Engine Keyword Arguments
    # =====================================================================

    engine_kwargs = build_engine_kwargs(args)

    # =====================================================================
    # Phase 3: Late Imports and Engine Creation
    # =====================================================================

    print(f"Loading model from {args.model}...")

    from babyvllm.engine.async_llm_engine import AsyncLLMEngine
    from babyvllm.entrypoints import api_server
    import uvicorn

    # Create the AsyncLLMEngine.
    # This is where:
    #   1. LLMEngine.__init__() → Config.__init__() validates model path
    #   2. Model weights are loaded onto GPU(s)
    #   3. Multiprocessing worker processes are spawned
    #   4. Tokenizer is loaded
    #   5. Scheduler + KV cache block manager are initialized
    #
    # If this fails (e.g., model not found, GPU OOM), the error propagates
    # naturally before the server starts.
    engine = AsyncLLMEngine(model=args.model, **engine_kwargs)
    print("Model loaded successfully.")

    # =====================================================================
    # Phase 4: Inject Engine into API Server Module
    # =====================================================================
    #
    # Module-level singleton pattern:
    #   api_server._engine is accessed by all endpoint handlers.
    #   Setting it here (before uvicorn.run()) ensures:
    #     - The engine is ready when the first request arrives
    #     - All requests use the same engine instance
    #     - The FastAPI lifespan can call engine.stop() on shutdown
    #
    # Why not pass engine as a FastAPI app.state dependency?
    #   app.state requires request-level dependency injection, which adds
    #   boilerplate in every endpoint. Module-level access is simpler and
    #   equally safe for single-worker deployments.
    # =====================================================================

    api_server._engine = engine

    # =====================================================================
    # Phase 5: Start HTTP Server
    # =====================================================================
    #
    # uvicorn.run() is a blocking call:
    #   - Starts the asyncio event loop
    #   - Registers the FastAPI app
    #   - Listens on the specified host:port
    #   - Runs until SIGTERM, SIGINT, or KeyboardInterrupt
    #
    # After uvicorn.run() returns:
    #   1. FastAPI lifespan triggers shutdown → engine.stop() clears background loop
    #   2. atexit handlers run → LLMEngine.exit() joins worker processes
    #   3. Python process exits
    # =====================================================================

    # WHY: Read host/port from engine.engine.config instead of raw CLI args.
    # At this point engine_kwargs (which included host/port) has been piped
    # through LLMEngine.__init__ → Config.__init__ → Config.__post_init__
    # validation.  So engine.engine.config holds the validated, canonical
    # values.  Using raw args would bypass validation — e.g. if a bug let an
    # out-of-range port into args, it would only be caught here by uvicorn
    # (late failure), not by Config.__post_init__ (early failure).
    #
    # Example: args.port=99999 survives argparse (which only checks type=int),
    # but Config.__post_init__ raises AssertionError("port must be between
    # 1 and 65535") before the model is loaded.  Then this code never runs —
    # a fast failure rather than a slow failure after GPU allocation.
    print(f"Starting server on {engine.engine.config.host}:{engine.engine.config.port}...")
    uvicorn.run(
        api_server.app,
        host=engine.engine.config.host,
        port=engine.engine.config.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
