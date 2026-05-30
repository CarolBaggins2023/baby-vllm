from babyvllm.sampling_params import SamplingParams

# LLMEngine and AsyncLLMEngine are imported lazily via __getattr__
# to avoid triggering GPU/model dependencies during unit-test imports.
__all__ = ["SamplingParams", "LLMEngine", "AsyncLLMEngine"]


def __getattr__(name):
    if name == "LLMEngine":
        from babyvllm.engine.llm_engine import LLMEngine as _LLMEngine
        return _LLMEngine
    if name == "AsyncLLMEngine":
        from babyvllm.engine.async_llm_engine import AsyncLLMEngine as _AsyncLLMEngine
        return _AsyncLLMEngine
    raise AttributeError(f"module 'babyvllm' has no attribute '{name}'")
