import torch
import torch.nn as nn
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from utils import get_context


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
    def __init__(self, num_heads: int, head_dim: int, scale: float = 1.0, num_kv_heads: int = None):
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
        
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        context = get_context()
        k_cache = self.k_cache
        v_cache = self.v_cache
        
        # If `k_cache` and `v_cache` are empty (torch.tensor([])), cache is not enabled.
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
        if context.is_prefill:
            if context.block_tables is not None:
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(
                q, k, v,
                max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale, causal=True, block_table=context.block_tables,
            )
        else:
            o = flash_attn_with_kvcache(
                q.unsqueeze(1), k_cache, v_cache,
                cache_seqlens=context.seq_lens, block_table=context.block_tables,
                softmax_scale=self.scale, causal=True
            )
        return o