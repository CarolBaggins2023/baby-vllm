from __future__ import annotations

import torch
import contextvars
from dataclasses import dataclass

@dataclass
class Context:
    is_prefill: bool = False
    
    # `cu_seqlens_q` is the cumulative sequence lengths, considering both cached and uncached tokens.
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
    cu_seqlens_q: torch.Tensor|None = None
    # `max_seqlen_q` is the maximum sequence length, considering both cached and uncached tokens.
    # For example, in the above example, the longest sequence is Sequence 2, with length 5,
    # so `max_seqlen_q` is 5.
    max_seqlen_q: int = 0
    # `cu_seqlens_k` is the cumulative sequence lengths, considering only uncached tokens.
    # It has the same data structure as `cu_seqlens_q`.
    # In above example, suppose token 2 in Sequence 1 is cached, and token 5 and token 6 in Sequence 2 are cached,
    # then `cu_seqlens_k` will be [0, 2, 4, 7].
    cu_seqlens_k: torch.Tensor|None = None
    # `max_seqlen_k` is the maximum sequence length, considering only uncached tokens.
    max_seqlen_k: int = 0
    
    # `slot_mapping` maps token index to slot index in cache block. sequence <-> cache block
    # `block_tables` maps sequence index to cache block indexs. token <-> slot in cache block
    
    # 1-dimension tensor, with shape of (num_tokens,).
    # It maps token index to cache slot index and maps padded token to -1.
    # For example, there are token 0, token 1 and padded token 2 and padded token 3.
    # If token 0 is written at cache slot 0, token 1 is written at cache slot 1,
    # `slot_mapping` should be [0, 1, -1, -1].
    slot_mapping: torch.Tensor|None = None
    
    # 2-dimension tensor, with shape of (num_sequences, num_blocks_per_sequence).
    # It maps sequence index to cache block indexs.
    # For examples, if Sequence 0 use Cache Block 0 and Cache Block 1, Sequence 1 use Cache Block 2,
    # then `block_tables` should be [[0, 1], [2]].
    block_tables: torch.Tensor|None = None
    
    # 1-dimension tensor, with shape of (num_sequences,).
    # It records the number of handled tokens (prompt length in prefill,
    # or generated length in decode) in each sequence.
    # For example, if Sequence 0 has 5 tokens in prompt, Sequence 1 has 3 tokens in prompt,
    # then the `context_lens` is [5, 3] after prefilling.
    # After prefilling and before first decoding, `context_lens` is still [5, 3].
    # After first deocoding and before second decoding, `context_lens` becomes [6, 4].
    context_lens: torch.Tensor|None = None
    
# ---------------------------------------------------------------------------
#  To implement online service:
#    Replace module-level global mutable singleton with contextvars.ContextVar.
#
#  Problem Background (before refactoring):
#    The old code used a module-level global variable `_CONTEXT = Context()` to store attention metadata.
#    Each call to set_context() directly overwrites this global object.
#    When multiple requests execute concurrently, different model forward passes overwrite each other's attention metadata.
#    It leads to:
#      - cu_seqlens / block_tables / slot_mapping corruption
#      - KV cache written to wrong locations, producing random outputs or even CUDA illegal memory access
#
#  Solution:
#    Python 3.7+'s built-in contextvars module provides "context variables" (ContextVar),
#    where each asyncio Task has its own independent copy of the variable.
#
#  How it works:
#      ┌──────────────────────────────────────────────────────┐
#      │  Task A                    Task B                    │
#      │  set_context(seqs=[1,2])   set_context(seqs=[3])    │
#      │       ↓                         ↓                   │
#      │  _context_var:             _context_var:            │
#      │    {seqs: [1,2]}             {seqs: [3]}            │
#      │       ↓                         ↓                   │
#      │  get_context() → [1,2]     get_context() → [3]      │
#      └──────────────────────────────────────────────────────┘
#    Values written via set() by each Task are only visible to itself and child Tasks derived from it.
#    Tasks are completely isolated and do not interfere with each other.
#
#  Why not threading.local:
#    threading.local isolates by thread, but multiple asyncio coroutines may share the same thread.
#    In this case, threading.local cannot distinguish between different Tasks,
#    leading to the same conflicts as global variables.
#
#  Why not multiprocessing.Manager:
#    The model needs to read context at each inference step.
#    Manager's IPC latency (~microsecond level) would seriously slow down inference throughput.
#    ContextVar is a pure C-implemented thread/coroutine local storage with negligible overhead.
# ---------------------------------------------------------------------------
# ContextVar itself is a module-level shared "key", but each Task uses this key to open its own "locker".
# default=Context() ensures that in environments where set() has not been called explicitly
# (e.g., single-threaded offline inference), get() returns a clean default Context, behaving exactly as before refactoring.
_context_var: contextvars.ContextVar[Context] = contextvars.ContextVar(
    'context',
    default=Context(),
)

def get_context() -> Context:
    """
    Get the Context object for the current context.

    Equivalent to reading the current Task's private attention metadata.
    In single-threaded offline inference, since set() is never called,
    each get() returns default=Context(), behaving exactly as
    the global singleton before refactoring.

    Call chain example (decode phase):
      model_runner.prepare_decode()             -- set_context(... current batch)
      model_runner.run_model()                  -- graph.replay()
        └─ attention.py:forward()               -- get_context().block_tables
           embedding_head.py:forward()           -- get_context().context_lens
      model_runner.run()                        -- reset_context()
    """
    return _context_var.get()

def reset_context():
    """
    Reset the current context to a brand new empty Context.

    Typically called after model forward pass completes to ensure the context is
    clean before the next step() begins, preventing residual data from the previous
    set_context() (e.g., slot_mapping) from being incorrectly reused.
    """
    _context_var.set(Context())

def set_context(
    is_prefill: bool,
    cu_seqlens_q: torch.Tensor | None = None,
    max_seqlen_q: int = 0,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_k: int = 0,
    slot_mapping: torch.Tensor | None = None,
    block_tables: torch.Tensor | None = None,
    context_lens: torch.Tensor | None = None,
):
    """
    Set all attention metadata required for the current context.

    ContextVar.set() only affects the current Task (and child Tasks derived from it),
    and does not pollute contexts of other concurrent requests.
    """
    _context_var.set(Context(
        is_prefill,
        cu_seqlens_q,
        max_seqlen_q,
        cu_seqlens_k,
        max_seqlen_k,
        slot_mapping,
        block_tables,
        context_lens,
    ))
