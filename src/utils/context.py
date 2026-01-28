import torch
from dataclasses import dataclass

@dataclass
class Context:
    is_prefill: bool = False
    
    # `cu_seqlens_q` is the cumulative sequence lengths of the queries.
    # For example, suppose there are 3 sequences, with length 2, 3, and 5:
    # Sequence 0: [token 0, token 1]
    # Sequence 1: [token 2, token 3, token 4]
    # Sequence 2: [token 5, token 6, token 7, token 8, token 9]
    # Then `cu_seqlens_q` will be [0, 2, 5, 10],
    # where `cu_seqlens_q[0]` denotes the begin index of Sequence 0,
    # `cu_seqlens_q[1]` denotes the end index of Sequence 0 and the begin index of Sequence 1,
    # `cu_seqlens_q[2]` denotes the end index of Sequence 1 and the begin index of Sequence 2,
    # `cu_seqlens_q[3]` denotes the end index of Sequence 2.
    # Therefore, based on `cu_seqlens_q`, we can get:
    # (1) The begin index of each sequence: `cu_seqlens_q[:-1]`
    # (2) The end index of each sequence: `cu_seqlens_q[1:]`
    cu_seqlens_q: torch.Tensor = None
    # `max_seqlen_q` is the maximum sequence length of the queries.
    # For example, in the above example, the longest sequence is Sequence 2, with length 5,
    # so `max_seqlen_q` is 5.
    max_seqlen_q: int = 0
    # `cu_seqlens_k` is the cumulative sequence lengths of the keys and values.
    # It has the same data structure as `cu_seqlens_q`.
    cu_seqlens_k: torch.Tensor = None
    # `max_seqlen_k` is the maximum sequence length of the keys and values.
    max_seqlen_k: int = 0
    
_context = Context()

def get_context() -> Context:
    return _context

def reset_context():
    global _context
    _context = Context()

def set_context(
    is_prefill,
    cu_seqlens_q = None,
    max_seqlen_q = 0,
    cu_seqlens_k = None,
    max_seqlen_k = 0,
):
    global _context
    _context = Context(
        is_prefill,
        cu_seqlens_q,
        max_seqlen_q,
        cu_seqlens_k,
        max_seqlen_k
    )