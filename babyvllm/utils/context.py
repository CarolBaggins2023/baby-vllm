import torch
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

    # ============================================================
    # Fields for mixed batch reordering.
    # ============================================================
    # After reordering, Decode sequences (q_len=1) are placed before Prefill sequences (q_len>1).
    # The following two fields record the "split point" in the merged token array.

    # Number of Decode sequences in the batch (those with chunk_size=1).
    # Since each Decode sequence contributes exactly 1 token, this equals num_decode_tokens.
    # For example, if the batch has 3 decode seqs and 2 prefill seqs:
    #   seqs (after reorder): [D0, D1, D2, P0, P1]
    #   num_decode_seqs = 3
    num_decode_seqs: int = 0
    # Number of Decode tokens in the merged input (same as num_decode_seqs).
    # This marks the index in the merged q/k/v tensors where Prefill tokens begin.
    # For example, if q has shape (307, ...):
    #   q[:3]   -> Decode tokens (D0, D1, D2, each q_len=1)
    #   q[3:307] -> Prefill tokens (P0 has q_len=50, P1 has q_len=254)
    num_decode_tokens: int = 0
    
_CONTEXT = Context()

def get_context() -> Context:
    return _CONTEXT

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()

def set_context(
    is_prefill,
    cu_seqlens_q = None,
    max_seqlen_q = 0,
    cu_seqlens_k = None,
    max_seqlen_k = 0,
    slot_mapping=None,
    block_tables=None,
    context_lens=None,
    num_decode_seqs: int = 0,
    num_decode_tokens: int = 0,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill,
        cu_seqlens_q,
        max_seqlen_q,
        cu_seqlens_k,
        max_seqlen_k,
        slot_mapping,
        block_tables,
        context_lens,
        num_decode_seqs,
        num_decode_tokens,
    )