__all__ = ["LLMEngine"]

def __getattr__(name):
    if name == "LLMEngine":
        from babyvllm.engine.llm_engine import LLMEngine
        return LLMEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
