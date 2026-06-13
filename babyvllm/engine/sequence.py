from __future__ import annotations

from enum import Enum, auto
from itertools import count
from copy import copy
import math
from typing import Any

from babyvllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

class Sequence:
    """Sequence of tokens. It also records the related cached blocks.
    """
    
    # shared class property
    # maximum number of tokens in a sequence
    block_size = 256
    # global index generator for sequences
    # It will generate increment unique sequence id.
    counter = count()
    
    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        # unique sequence id
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        
        self.token_ids = copy(token_ids)
        self.last_token = self.token_ids[-1] if self.token_ids else None
        self.num_tokens = len(self.token_ids)
        # When the sequence is created, it is filled with prompt tokens.
        self.num_prompt_tokens = len(self.token_ids)
        
        self.num_cached_tokens = 0
        self.block_table = []
        
        # Sampling parameters for the sequence.
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.max_model_length = sampling_params.max_model_length
    
    def __len__(self):
        return self.num_tokens
    
    def __getitem__(self, idx):
        return self.token_ids[idx]
    
    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED
    
    @property
    def num_completion_tokens(self):
        return self.num_tokens-self.num_prompt_tokens
    
    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]
    
    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]
    
    @property
    def num_blocks(self):
        return math.ceil(self.num_tokens/self.block_size)

    @property    
    def num_cached_blocks(self):
        return math.ceil(self.num_cached_tokens/self.block_size)
    
    @property
    def last_block_num_tokens(self):
        """ Calculate the number of tokens in the last block. """
        return self.num_tokens-(self.num_blocks-1)*self.block_size
    
    def block(self, i: int):
        """ Get the i-th block. """
        assert 0 <= i < self.num_blocks, f"Block index {i} out of range [0, {self.num_blocks})"
        # In python, slice logic automatically clips the indices to the length of the list.
        # So, "(i+1)*self.block_size" just works.
        return self.token_ids[i*self.block_size:(i+1)*self.block_size]
    
    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def prefill_uncached_token_count(self) -> int:
        return len(self.token_ids)-self.num_cached_tokens

    def validate_prefill_state(self) -> int:
        num_uncached_tokens = self.prefill_uncached_token_count()
        if num_uncached_tokens <= 0:
            seq_id = getattr(self, "seq_id", "?")
            raise ValueError(
                f"Sequence {seq_id} has no uncached query tokens for prefill "
                f"(token_ids={len(self.token_ids)}, "
                f"num_cached_tokens={self.num_cached_tokens}). "
                "Prefix cache must leave at least one token uncached."
            )
        return num_uncached_tokens

    def to_worker_state(self, is_prefill: bool) -> dict[str, Any]:
        """Build a phase-aware, pickle-friendly state for TP worker ranks."""

        return {
            "version": 1,
            "phase": "prefill" if is_prefill else "decode",
            "seq_id": self.seq_id,
            "status": self.status.name,
            "num_tokens": self.num_tokens,
            "num_prompt_tokens": self.num_prompt_tokens,
            "num_cached_tokens": self.num_cached_tokens,
            "block_table": list(self.block_table),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "ignore_eos": self.ignore_eos,
            "max_model_length": self.max_model_length,
            "last_token": self.last_token,
            "token_ids": list(self.token_ids) if is_prefill else None,
        }

    @classmethod
    def from_worker_state(cls, state: dict[str, Any]) -> "Sequence":
        phase = state.get("phase")
        if phase == "prefill":
            token_ids = list(state["token_ids"])
            if len(token_ids) != state["num_tokens"]:
                raise ValueError(
                    "Prefill sequence worker state must include full token_ids "
                    f"(got {len(token_ids)}, expected {state['num_tokens']})."
                )
        elif phase == "decode":
            token_ids = [state["last_token"]]
        else:
            raise ValueError(f"Unknown sequence worker-state phase: {phase!r}")

        seq = cls.__new__(cls)
        seq.seq_id = state["seq_id"]
        status = state.get("status", "WAITING")
        seq.status = SequenceStatus[status] if isinstance(status, str) else status
        seq.token_ids = token_ids
        seq.last_token = state["last_token"]
        seq.num_tokens = state["num_tokens"]
        seq.num_prompt_tokens = state["num_prompt_tokens"]
        seq.num_cached_tokens = state["num_cached_tokens"]
        seq.block_table = list(state["block_table"])
        seq.temperature = state["temperature"]
        seq.max_tokens = state["max_tokens"]
        seq.ignore_eos = state["ignore_eos"]
        seq.max_model_length = state["max_model_length"]
        return seq
    
    def __getstate__(self):
        return self.to_worker_state(is_prefill=True)
    
    def __setstate__(self, state):
        if isinstance(state, dict):
            restored = self.from_worker_state(state)
            self.__dict__.update(restored.__dict__)
            return

        # Backward compatibility for old tuple pickles.
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.token_ids = [state[-1]]
        self.last_token = self.token_ids[-1] if self.token_ids else None
        self.temperature = 1.0
        self.max_tokens = 64
        self.ignore_eos = False
        self.max_model_length = None
    
