import torch
import contextvars
from dataclasses import dataclass

@dataclass
class Context:
    is_prefill: bool = False

    # `cu_seqlens_q` is the cumulative length of query tokens in the current
    # physical forward. For prefill/chunked prefill, each sequence contributes
    # its scheduled chunk_size. For decode, q metadata is unused by
    # flash_attn_with_kvcache.
    # For example, if 3 sequences contribute current chunks of length 2, 3, and 5:
    # Sequence 0 query tokens: [token 0, token 1]
    # Sequence 1 query tokens: [token 2, token 3, token 4]
    # Sequence 2 query tokens: [token 5, token 6, token 7, token 8, token 9]
    # Then `cu_seqlens_q` is [0, 2, 5, 10],
    # where `cu_seqlens_q[0]` denotes the begin index of Sequence 0,
    # `cu_seqlens_q[1]` denotes the end index of Sequence 0 and the begin index of Sequence 1,
    # `cu_seqlens_q[2]` denotes the end index of Sequence 1 and the begin index of Sequence 2,
    # `cu_seqlens_q[3]` denotes the end index of Sequence 2.
    # Therefore, based on `cu_seqlens_q`, we can get:
    # (1) The begin index of each sequence: `cu_seqlens_q[:-1]`
    # (2) The end index of each sequence: `cu_seqlens_q[1:]`
    cu_seqlens_q: torch.Tensor|None = None
    # `max_seqlen_q` is the maximum current query chunk length.
    # For example, in the above example, the longest sequence is Sequence 2, with length 5,
    # so `max_seqlen_q` is 5.
    max_seqlen_q: int = 0
    # `cu_seqlens_k` is the cumulative visible context length for each
    # sequence in the current forward: previous cached tokens plus the current
    # chunk. It has the same prefix-sum structure as `cu_seqlens_q`.
    # In the above example, if Sequence 1 has 1 previous token and Sequence 2
    # has 2 previous tokens, then k lengths are [2, 4, 7] and
    # `cu_seqlens_k` is [0, 2, 6, 13].
    cu_seqlens_k: torch.Tensor|None = None
    # `max_seqlen_k` is the maximum visible context length.
    max_seqlen_k: int = 0

    # `slot_mapping` maps each current input token to an absolute KV-cache slot.
    # `block_tables` maps each sequence to the physical KV-cache blocks that
    # store its history.

    # 1-dimension tensor, with shape of (num_tokens,).
    # It maps token index to cache slot index and maps padded token to -1.
    # For example, there are token 0, token 1 and padded token 2 and padded token 3.
    # If token 0 is written at cache slot 0, token 1 is written at cache slot 1,
    # `slot_mapping` should be [0, 1, -1, -1].
    slot_mapping: torch.Tensor|None = None

    # 2-dimension tensor, with shape of (num_sequences, num_blocks_per_sequence).
    # It maps sequence index to cache block indexes.
    # For examples, if Sequence 0 use Cache Block 0 and Cache Block 1, Sequence 1 use Cache Block 2,
    # then `block_tables` should be [[0, 1], [2]].
    block_tables: torch.Tensor|None = None

    # 1-dimension tensor, with shape of (num_sequences,).
    # It records the visible context length for each sequence after the current
    # chunk has been written to KV cache. This is prompt tokens plus generated
    # tokens, not just the number of completion tokens.
    # For example, if Sequence 0 has 5 prompt tokens and Sequence 1 has 3 prompt
    # tokens, then `context_lens` is [5, 3] after prefilling.
    # After the first decode token for each sequence, `context_lens` becomes [6, 4].
    context_lens: torch.Tensor|None = None
    
# ---------------------------------------------------------------------------
# Attention metadata is stored in a ContextVar rather than a module-level
# mutable singleton. Each asyncio Task gets an isolated Context, so concurrent
# requests cannot overwrite each other's cu_seqlens, block tables, slot mapping,
# or KV-cache metadata.
#
#  Why ContextVar:
#    Python 3.7+'s built-in contextvars module provides task-local values.
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
#    Values written via set() by each Task are only visible to itself and
#    child Tasks derived from it. Tasks are completely isolated and do not
#    interfere with each other.
#
#  Why not threading.local:
#    threading.local isolates by thread, but multiple asyncio coroutines may
#    share the same thread. In this case, threading.local cannot distinguish
#    between different Tasks, leading to the same conflicts as global variables.
#
#  Why not multiprocessing.Manager:
#    The model needs to read context at each inference step. Manager's IPC
#    latency (~microsecond level) would seriously slow down inference throughput.
#    ContextVar is a pure C-implemented thread/coroutine local storage with
#    negligible overhead.
# ---------------------------------------------------------------------------
# ContextVar itself is a module-level shared "key", but each Task gets its own
# value. default=Context() keeps single-threaded/offline inference working even
# when set_context() has not been called explicitly.
_context_var: contextvars.ContextVar[Context] = contextvars.ContextVar(
    'context', default=Context()
)

def get_context() -> Context:
    return _context_var.get()

def reset_context():
    _context_var.set(Context())

def set_context(
    is_prefill: bool,
    cu_seqlens_q = None,
    max_seqlen_q: int = 0,
    cu_seqlens_k = None,
    max_seqlen_k: int = 0,
    slot_mapping = None,
    block_tables = None,
    context_lens = None,
):
    _context_var.set(Context(
        is_prefill,
        cu_seqlens_q, max_seqlen_q,
        cu_seqlens_k, max_seqlen_k,
        slot_mapping, block_tables, context_lens,
    ))
    
