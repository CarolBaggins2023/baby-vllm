# baby-vllm API Server Entrypoints Package
# Contains the FastAPI-based OpenAI-compatible REST API server and CLI launcher.
#
# Submodules:
#   api_server.py — FastAPI application with /v1/completions, /v1/chat/completions,
#                    /v1/models, /health endpoints
#   cli.py        — Argument parser and uvicorn launcher (babyvllm-server command)
