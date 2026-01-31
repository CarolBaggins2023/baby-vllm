import torch
import torch.nn as nn
import torch.distributed as dist

from layers import *

class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int|None = None,
        scale: float = 1.0,
        num_kv_heads: int|None = None,
        qkv_bias: bool = False,
        base: int = 10000,
        max_position: int = 16384,
    ):
        super().__init__()
        
        self.tp_size = dist.get_world_size()
        
        # Each GPU deals with different attention heads.
        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads//self.tp_size
        self.total_num_kv_heads = num_kv_heads if num_kv_heads is not None else self.total_num_heads
        self.num_kv_heads = self.total_num_kv_heads//self.tp_size
        
        self.head_dim = head_dim if head_dim is not None else hidden_size//self.tp_size
        self.q_size = self.head_dim*self.num_heads
        self.kv_size = self.head_dim*self.num_kv_heads
        self.qkv_bias = qkv_bias
        
        self.qkv_projection = QKVColumnParallelLinear(
            input_size=self.head_dim*self.total_num_heads,
            head_size=self.head_dim,
            num_heads=self.total_num_heads,
            num_kv_heads=self.total_num_kv_heads,
            bias=qkv_bias,
        )
        
        self.rms_norm = LayerNorm(torch.ones(head_dim))
        
        self.rotary_emb = RotaryEmbedding(
            base=base,
            rotary_dim=self.head_dim,
            max_position=max_position,
        )
        
        self.attention = Attention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scale=scale,
            num_kv_heads=self.num_kv_heads,
        )
        
        self.out_projection = RowParallelLinear(
            input_size=self.head_dim*self.total_num_heads,
            output_size=hidden_size,
            bias=qkv_bias,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # `x` is replicated across all GPUs.
        # x shape: (batch_size, seq_len, hidden_size)
        # positions shape: (batch_size, seq_len)
        
        # Column Parallel Linear -> Scaled Dot-Product (GPU 0)         -> Row Parallel Linear
        #                        -> Scaled Dot-Product (GPU 1)         ->
        #                        -> ...                                ->
        #                        -> Scaled Dot-Product (GPU tp_size-1) ->
        
        # ===== (1) QKV Projection and Split (Sharding Happens) =====
        # `QKVColumnParallelLinear` returns the shard of qkv.
        # qkv shape: (batch_size, seq_len, head_dim*(num_heads+2*num_kv_heads))
        qkv = self.qkv_projection(x)
        
        # Split qkv into q, k, and v, according to their sizes.
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # Split heads for q, k, and v.
        # Handle both varlen mode (2D) and batched mode (3D).
        if q.dim() == 2:
            # Varlen Mode
            # q shape: (total_tokens, q_size) -> (total_tokens, num_heads, head_dim)
            q = q.view(-1, self.num_heads, self.head_dim)
            # k, v shape: (total_tokens, kv_size) -> (total_tokens, num_kv_heads, head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)
        else:
            # Batched Mode
            batch_size, seq_len, _, _ = q.shape
            # q shape: (batch_size, seq_len, q_size) -> (batch_size, seq_len, num_heads, head_dim)
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
            # k, v shape: (batch_size, seq_len, kv_size) -> (batch_size, seq_len, num_kv_heads, head_dim)
            k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # ===== (2) Normalize Q and K and Apply Rotary Embedding =====
        # When computing attention weights, there is softmax(q.dot(k)/sqrt(head_dim)).
        # So, if there is big number in q.dot(k), the softmax will be nan.
        # To avoid this, we normalize q and k.
        if self.qkv_bias is False:
            q = self.rms_norm(q)
            k = self.rms_norm(k)
        
        # Apply rotary embedding to q and k.
        q, k = self.rotary_emb(positions, q, k)
        
        # ===== (4) Scaled Dot-Product Attention =====
        # o shape: (batch_size*seq_len, num_heads, head_dim)
        o = self.attention(q, k, v)
        
        # ===== (5) Output Projection (Communication Happens by All Reduce) =====
        # o shape: (batch_size*seq_len, hidden_size)
        o = self.out_projection(o)
        
        return o

class Qwen3MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = True,
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size]*2,
            bias=bias,
        )
        self.activation = SiluAndMul()
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gate_up_proj(x)
        x = self.activation(x)
        x = self.down_proj(x)
        return x

class Qwen3DecoderLayer(nn.Module):
    pass

class Qwen3Model(nn.Module):
    pass

class Qwen3ForCausalLM(nn.Module):
    pass
