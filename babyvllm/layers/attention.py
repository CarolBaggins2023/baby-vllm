import torch
import torch.nn as nn
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from babyvllm.utils import Context, get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    """
    Store the key and value tensors into the pre-allocated key and value caches.
    
    Args:
        key_ptr (torch.Tensor): The pointer to key tensor, with shape (num_tokens, num_kv_heads, head_dim).
        key_stride (int): The stride of key tensor.
        value_ptr (torch.Tensor): The pointer to value tensor, with shape (num_tokens, num_kv_heads, head_dim).
        value_stride (int): The stride of value tensor.
        k_cache_ptr (torch.Tensor): The pointer to pre-allocated key cache, with shape (num_blocks, block_size, num_kv_heads, head_dim).
        v_cache_ptr (torch.Tensor): The pointer to pre-allocated value cache, with shape (num_blocks, block_size, num_kv_heads, head_dim).
        slot_mapping_ptr (torch.Tensor): Map token index to cache slot index.
        D (int): The dimension of the key and value tensors.
        
        key_ptr and value_ptr point to what we want to store.
        k_cache_ptr and v_cache_ptr point to where we want to store.
    """
    # One program handles one token.
    # The program index, which is also the token index.
    token_idx = tl.program_id(0)
    # The index of the cache slot where we want to store the key and value.
    slot = tl.load(slot_mapping_ptr+token_idx)
    # If the slot is -1, it means the token is padding, we should skip it.
    if slot == -1:
        return
    
    # `tl.arange(0, D)` creates a vector [0, ..., D-1]
    key_offsets = (token_idx*key_stride+ # skip keys of previous tokens
                   tl.arange(0, D))
    value_offsets = (token_idx*value_stride+ # skip values of previous tokens
                     tl.arange(0, D))
    # Load the key and value of this token.
    key = tl.load(key_ptr+key_offsets)
    value = tl.load(value_ptr+value_offsets)
    
    cache_offset = (slot*D+ # skip caches of previous tokens
                    tl.arange(0, D))
    # Store the key and value of this token into the cache.
    tl.store(k_cache_ptr+cache_offset, key)
    tl.store(v_cache_ptr+cache_offset, value)

def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """
    Store the key and value tensors into the pre-allocated caches.
    
    Args:
        key (torch.Tensor): The key tensor computed by attention, with shape (num_tokens, num_kv_heads, head_dim).
        value (torch.Tensor): The value tensor computed by attention, with shape (num_tokens, num_kv_heads, head_dim).
        k_cache (torch.Tensor): The pre-allocated key cache, with shape (num_blocks, block_size, num_kv_heads, head_dim).
        v_cache (torch.Tensor): The pre-allocated value cache, with shape (num_blocks, block_size, num_kv_heads, head_dim).
        slot_mapping (torch.Tensor): Mapping token index to cache index, with shape (num_tokens,).
    """
    num_tokens, num_kv_heads, head_dim = key.shape
    
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()
    
    # Check memory layout of variables to ensure we can access them in a expected way.
    
    # In the kernel, we access key and value in the way:
    # key_offsets = (token_idx*key_stride+tl.arange(0, D)),
    # key = tl.load(key_ptr+key_offsets),
    # where `tl.arange(0, D)` is continuously increasing.
    # So, the memory layout of key and value should be:
    # (1) `head_dim` elements within a head are continuously stored.
    # (2) `num_kv_heads` heads are continuously stored.
    # For example, head 0 has key 0, key 1, key 2, head 1 has key 3, key 4, key 5.
    # In memory, they should be stored as:
    #   0        1        2        3        4        5
    # key 0    key 1    key 2    key 3    key 4    key 5
    # If head 0 is stored as:
    #   0        1      x      2    
    # key 0    key 1    x    key 2  
    # or if head 0 and head 1 is stored as:
    #   0        1        2     x     3        4        5
    # key 0    key 1    key 2   x   key 3    key 4    key 5
    # `key_ptr+key_offsets` can not cover all keys of head 0 and head 1, because there are some padding elements x.
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim*1 and value.stride(1) == head_dim*1
    
    # Similarly, elements in each slot of cache should be continuously stored.
    assert k_cache.stride(1) == num_kv_heads*head_dim and v_cache.stride(1) == num_kv_heads*head_dim
    
    # `slot_mapping` maps token index to cache slot index, so it should have the same number of elements as key and value.
    assert slot_mapping.numel() == num_tokens
    
    # One program handles one token.
    store_kvcache_kernel[(num_tokens,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D=num_kv_heads*head_dim)

class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int = None
    ):
        """
        Args:
            num_heads (int): The number of query attention heads.
            head_dim (int): The dimension of each attention head.
            scale (float, optional): The scaling factor for the attention scores, usually set to 1/sqrt(head_dim).
            num_kv_heads (int, optional): The number of key-value attention heads. If None, defaults to num_heads.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        
        # `k_cache` and `v_cache` will refer to the cache pool which is allocated in model runner.
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Attention forward pass with Decode/Prefill split dispatch.
        """

        # After batch reordering, Decode sequences (q_len=1) come first,
        # followed by Prefill sequences (q_len>1) in the merged q/k/v tensors.
        #
        # This method splits the attention computation at the boundary so each
        # group uses its optimal FlashAttention operator:
        #   - Decode tokens  -> flash_attn_with_kvcache  (optimized for q_len=1)
        #   - Prefill tokens -> flash_attn_varlen_func    (handles variable q_len>1)
        #
        # Edge cases:
        #   - Pure Decode batch  (no Prefill): only the Decode branch fires.
        #   - Pure Prefill batch (no Decode):  only the Prefill branch fires.
        #   - Mixed batch:                     both branches fire, results concatenated.
        #
        # Example: batch = [D0, D1, D2, P0, P1] where each Dx has 1 token,
        # P0 has 50 tokens, P1 has 254 tokens.
        #   num_decode_tokens  = 3   <-- split point
        #   num_prefill_tokens = 304
        #   q[:3]     -> Decode tokens  (D0, D1, D2), dispatched to flash_attn_with_kvcache
        #   q[3:307]  -> Prefill tokens (P0: 50, P1: 254), dispatched to flash_attn_varlen_func
        #   o = cat([o_decode, o_prefill], dim=0)  -- shape: (307, num_heads, head_dim)
        
        context: Context = get_context()
        k_cache = self.k_cache
        v_cache = self.v_cache

        # ============================================
        # 1. Separation of storage and computation.
        # ============================================
        # Store KV cache for ALL tokens.
        # Whether a token belongs to a Decode or Prefill sequence, its K/V must be
        # persisted into `k_cache`/`v_cache` for future queries to attend to.
        # `context.slot_mapping` tells each token where in the cache pool to write.
        if k_cache.numel() and v_cache.numel() and context.slot_mapping is not None:
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        # ============================================
        # 2. Split attention calculation.
        # ============================================
        # After batch reordering, tokens [0, num_decode_tokens) are Decode tokens,
        # and tokens [num_decode_tokens, total_tokens) are Prefill tokens.
        # We no longer `use context.is_prefill` for dispatch — instead we check
        # the actual token counts, which allows mixed batches to use both operators.

        total_tokens = q.shape[0]
        num_decode_tokens = context.num_decode_tokens
        num_prefill_tokens = total_tokens - num_decode_tokens
        num_decode_seqs = context.num_decode_seqs

        outputs = []  # Collect Decode and/or Prefill outputs, then concatenate.

        # --- Branch 1: Decode tokens (q_len == 1 for each sequence) ---
        if num_decode_tokens > 0:
            # Slice out the Decode portion from the merged q tensor.
            # Shape: (num_decode_tokens, num_heads, head_dim)
            #
            # Example: q.shape = (307, 32, 128), num_decode_tokens = 3
            #   q_decode = q[:3]  -> shape (3, 32, 128), one row per Decode seq.
            q_decode = q[:num_decode_tokens]

            # Slice related metadata to match the Decode sub-batch.
            # context.context_lens and context.block_tables are per-sequence arrays
            # (not per-token), and Decode seqs are the first `num_decode_seqs` entries.
            #
            # Example: if context.context_lens = [1001, 501, 301, 256, 150]
            #   (D0:1001, D1:501, D2:301, P0:256, P1:150)
            #   then decode_context_lens = [1001, 501, 301]
            decode_context_lens = context.context_lens[:num_decode_seqs]
            decode_block_tables = (
                context.block_tables[:num_decode_seqs]
                if context.block_tables is not None else None
            )

            # flash_attn_with_kvcache requires q shape (batch, 1, num_heads, head_dim).
            # We unsqueeze dim=1 to add the seq_len dimension (=1 for Decode).
            o_decode = flash_attn_with_kvcache(
                q_decode.unsqueeze(1),
                k_cache, v_cache,
                cache_seqlens=decode_context_lens,
                block_table=decode_block_tables,
                softmax_scale=self.scale,
                causal=True,
            )
            # Remove the artificial seq_len dimension.
            # o_decode shape: (num_decode_tokens, 1, num_heads, head_dim)
            #              -> (num_decode_tokens, num_heads, head_dim)
            o_decode = o_decode.squeeze(1)
            outputs.append(o_decode)

        # --- Branch 2: Prefill tokens (q_len > 1 for each sequence) ---
        if num_prefill_tokens > 0:
            # Slice out the Prefill portion from the merged q tensor.
            # Shape: (num_prefill_tokens, num_heads, head_dim)
            #
            # Example: q.shape = (307, 32, 128), num_decode_tokens = 3
            #   q_prefill = q[3:]  -> shape (304, 32, 128)
            #   These 304 tokens belong to P0 (50 tokens) and P1 (254 tokens).
            q_prefill = q[num_decode_tokens:]

            # ---------------------------------------------------------------
            # Construct sub-metadata for the Prefill portion.
            # Since we sliced out the Decode tokens, all cumulative indices
            # (cu_seqlens_q, cu_seqlens_k) must be shifted so they start from 0
            # for the first Prefill token.
            # ---------------------------------------------------------------

            # --- cu_seqlens_q ---
            # context.cu_seqlens_q covers ALL sequences (Decode + Prefill).
            # We take the tail starting from num_decode_seqs, then shift by the
            # offset value (which is the starting index of Prefill tokens in q).
            # The offset is pre-computed on the CPU side in prepare_forward() to
            # avoid GPU-CPU sync here.
            #
            # Example:
            #   Batch after reorder: [D0(1), D1(1), D2(1), P0(50), P1(254)]
            #   context.cu_seqlens_q = [0, 1, 2, 3, 53, 307]
            #   num_decode_seqs = 3
            #   cu_seqlens_q_offset = 3  <-- q index where Prefill begins
            #   Sliced: cu_seqlens_q[3:]  = [3, 53, 307]
            #   Shifted: [3-3, 53-3, 307-3] = [0, 50, 304]
            #   Interpreting [0, 50, 304]:
            #     P0 spans q_prefill[0:50], P1 spans q_prefill[50:304]
            prefill_cu_seqlens_q = context.cu_seqlens_q[num_decode_seqs:] - context.cu_seqlens_q_offset
            prefill_max_seqlen_q = context.prefill_max_seqlen_q

            # --- cu_seqlens_k ---
            # Same slicing and shifting logic as cu_seqlens_q.
            # context.cu_seqlens_k tracks the total key length (cached + new) per sequence.
            # The offset is pre-computed on the CPU side to avoid GPU-CPU sync.
            #
            # Example:
            #   context.cu_seqlens_k = [0, 1001, 1502, 1803, 2059, 2213]
            #   (D0 attended to 1001 keys, D1 to 501, D2 to 301, P0 chunk to 256, P1 chunk to 154)
            #   cu_seqlens_k_offset = 1803
            #   Sliced: [1803, 2059, 2213]
            #   Shifted: [0, 256, 410]  -- P0 key span is 256, P1 key span is 154
            prefill_cu_seqlens_k = context.cu_seqlens_k[num_decode_seqs:] - context.cu_seqlens_k_offset
            prefill_max_seqlen_k = context.prefill_max_seqlen_k

            # --- block_tables ---
            # Slice to only the Prefill sequences' block tables.
            # Shape: (num_prefill_seqs, max_blocks_per_seq)
            prefill_block_tables = (
                context.block_tables[num_decode_seqs:]
                if context.block_tables is not None else None
            )

            # --- attn_k / attn_v ---
            # When block_tables is provided, the FlashAttention operator reads
            # historical KV from the physical cache (k_cache/v_cache) using the
            # block table as a lookup. Otherwise, we fall back to the local k/v
            # tensors (sliced to the Prefill portion only).
            if prefill_block_tables is not None:
                attn_k = k_cache
                attn_v = v_cache
            else:
                # KV cache is not enabled; use the incoming local k/v.
                # Must slice to Prefill tokens only, matching q_prefill.
                attn_k = k[num_decode_tokens:]
                attn_v = v[num_decode_tokens:]

            o_prefill = flash_attn_varlen_func(
                q_prefill, attn_k, attn_v,
                max_seqlen_q=prefill_max_seqlen_q,
                cu_seqlens_q=prefill_cu_seqlens_q,
                max_seqlen_k=prefill_max_seqlen_k,
                cu_seqlens_k=prefill_cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=prefill_block_tables,
            )
            outputs.append(o_prefill)

        # ============================================
        # 3. Merge: concatenate Decode and Prefill outputs
        # ============================================
        # The concatenation order matches the reordered input order:
        #   o = [o_decode (Decode tokens), o_prefill (Prefill tokens)]
        # This is the same order that downstream per-position operations
        # (output projection, residual, FFN) will process.
        # The final token_ids are unsorted back to original sequence order
        # in model_runner.run() using self._sort_permutation.
        if len(outputs) == 2:
            o = torch.cat(outputs, dim=0)
        else:
            # Pure Decode or Pure Prefill — only one branch produced output.
            o = outputs[0]

        return o
