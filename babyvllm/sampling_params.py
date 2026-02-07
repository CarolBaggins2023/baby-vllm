from dataclasses import dataclass

@dataclass
class SamplingParams:
    temperature: float = 1.0
    # maximum number of completion tokens
    max_tokens: int = 64
    ignore_eos: bool = False
    # maximum number of total tokens (prompt+completion)
    max_model_length: int|None = None
    
    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
