from __future__ import annotations

import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import Qwen3Config

from babyvllm.layers import *
from babyvllm.utils import get_context


def _tp_checked_divide(
    *,
    field_name: str,
    value: int,
    tp_size: int,
    local_name: str,
    extra: str | None = None,
) -> int:
    if value % tp_size != 0:
        message = (
            f"{field_name}={value} must be divisible by tensor_parallel_size={tp_size}."
        )
        if extra is not None:
            message = f"{message} {extra}"
        raise ValueError(message)
    local_value = value//tp_size
    if local_value <= 0:
        raise ValueError(
            f"tensor_parallel_size={tp_size} produces zero {local_name} "
            f"from {field_name}={value}."
        )
    return local_value


class Qwen3Attention(nn.Module):
    """
    Attention Block in Qwen3.
    (1) QKV Projection and Split
    (2) Normalize Q and K and Apply Rotary Embedding
    (3) Scaled Dot-Product Attention
    (4) Output Projection
    """
    def __init__(self, config: Qwen3Config):
        super().__init__()
        
        self.tp_size = dist.get_world_size()
        
        # Each GPU deals with different attention heads.
        self.total_num_heads = config.num_attention_heads
        self.num_heads = _tp_checked_divide(
            field_name="num_attention_heads",
            value=self.total_num_heads,
            tp_size=self.tp_size,
            local_name="local attention heads",
        )
        self.total_num_kv_heads = config.num_key_value_heads if config.num_key_value_heads is not None else self.total_num_heads
        self.num_kv_heads = _tp_checked_divide(
            field_name="num_key_value_heads",
            value=self.total_num_kv_heads,
            tp_size=self.tp_size,
            local_name="local KV heads",
            extra="KV-head replication is not supported by baby-vllm-basic.",
        )
        
        self.head_dim = config.head_dim if config.head_dim is not None else config.hidden_size//config.num_attention_heads
        self.q_size = self.head_dim*self.num_heads
        self.kv_size = self.head_dim*self.num_kv_heads
        self.qkv_bias = config.attention_bias
        
        self.scale = self.head_dim**-0.5
        
        self.qkv_proj = QKVColumnParallelLinear(
            input_size=config.hidden_size,
            head_size=self.head_dim,
            num_heads=self.total_num_heads,
            num_kv_heads=self.total_num_kv_heads,
            bias=self.qkv_bias,
        )
        
        if not self.qkv_bias:
            self.q_norm = RMSNorm(
                hidden_size=self.head_dim,
                eps=config.rms_norm_eps,
            )
            self.k_norm = RMSNorm(
                hidden_size=self.head_dim,
                eps=config.rms_norm_eps,
            )
        
        self.rotary_emb = RotaryEmbedding(
            base=config.rope_parameters['rope_theta'],
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
        )
        
        self.attention = Attention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scale=self.scale,
            num_kv_heads=self.num_kv_heads,
        )
        
        self.o_proj = RowParallelLinear(
            input_size=self.head_dim*self.total_num_heads,
            output_size=config.hidden_size,
            bias=self.qkv_bias,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # `x` is replicated across all GPUs.
        # x shape: (total_tokens, hidden_size)
        # positions shape: (total_tokens,)
        
        # Column Parallel Linear -> Scaled Dot-Product (GPU 0)         -> Row Parallel Linear
        #                        -> Scaled Dot-Product (GPU 1)         ->
        #                        -> ...                                ->
        #                        -> Scaled Dot-Product (GPU tp_size-1) ->
        
        # ===== (1) QKV Projection and Split (Sharding Happens) =====
        # `QKVColumnParallelLinear` returns the shard of qkv.
        # qkv shape: (total_tokens, head_dim*(num_heads+2*num_kv_heads))
        qkv = self.qkv_proj(x)
        
        # Split qkv into q, k, and v, according to their sizes.
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # Split heads for q, k, and v.
        # Handle both varlen mode (2D) and batched mode (3D).
        # q shape: (total_tokens, q_size) -> (total_tokens, num_heads, head_dim)
        q = q.view(-1, self.num_heads, self.head_dim)
        # k, v shape: (total_tokens, kv_size) -> (total_tokens, num_kv_heads, head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        
        # ===== (2) Normalize Q and K and Apply Rotary Embedding =====
        # When computing attention weights, there is softmax(q.dot(k)/sqrt(head_dim)).
        # So, if there is big number in q.dot(k), the softmax will be nan.
        # To avoid this, we normalize q and k.
        if self.qkv_bias is False:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # Apply rotary embedding to q and k.
        q, k = self.rotary_emb(positions, q, k)
        
        # ===== (3) Scaled Dot-Product Attention =====
        # o shape: (total_tokens, num_heads, head_dim)
        o = self.attention(q, k, v)
        
        # ===== (4) Output Projection (Communication Happens by All Reduce) =====
        # Merge heads.
        # o shape: (total_tokens, num_heads, head_dim) -> (total_tokens, num_heads*head_dim)
        o = o.view(o.shape[0], -1)
        # o shape: (total_tokens, hidden_size)
        o = self.o_proj(o)
        
        return o

class Qwen3MLP(nn.Module):
    """
    MLP Block in Qwen3.
    (1) Gate-Up Projection
    (2) Activation
    (3) Down Projection
    """
    def __init__(self, config: Qwen3Config):
        super().__init__()
        _tp_checked_divide(
            field_name="intermediate_size",
            value=config.intermediate_size,
            tp_size=dist.get_world_size(),
            local_name="local MLP features",
        )
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[config.intermediate_size]*2,
            bias=False,
        )
        self.activation = SiluAndMul()
        self.down_proj = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=False,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gate_up_proj(x)
        x = self.activation(x)
        x = self.down_proj(x)
        return x

class Qwen3DecoderLayer(nn.Module):
    """
    Decoder Layer in Qwen3.
    (1) Input LayerNorm
    (2) Self-Attention
    (3) Post-Attention LayerNorm
    (4) MLP
    """
    def __init__(self, config: Qwen3Config):
        super().__init__()
        
        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.self_attn = Qwen3Attention(config)
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps
        )
        self.mlp = Qwen3MLP(config)
    
    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        residual: torch.Tensor|None = None,
    ) -> torch.Tensor:
        # TP rhythm inside one decoder layer:
        #   full hidden state replicated on all ranks
        #     -> ColumnParallel / MergedColumnParallel
        #     -> each rank gets a local shard
        #     -> local compute
        #     -> RowParallel
        #     -> all_reduce
        #     -> full hidden state replicated on all ranks
        #
        # Attention follows:
        #   QKVColumnParallelLinear -> local attention heads -> RowParallelLinear
        #
        # MLP follows:
        #   MergedColumnParallelLinear -> local activation -> RowParallelLinear
        #
        # This keeps residual, RMSNorm, and the next decoder layer seeing the
        # same hidden-state shape on every TP rank.

        # (1) Input LayerNorm
        if residual is not None:
            x, residual = self.input_layernorm(x, residual)
        else:
            x, residual = self.input_layernorm(x), x
            
        # (2) Self-Attention        
        x = self.self_attn(x, positions)
        
        # (3) Post-Attention LayerNorm
        x, residual = self.post_attention_layernorm(x, residual)
        
        # (4) MLP
        x = self.mlp(x)
        
        return x, residual

class Qwen3Model(nn.Module):
    """
    Qwen3 Model.
    (1) Embedding
    (2) Multiple Self-Attention and MLP
    (3) Final LayerNorm
    """
    def __init__(self, config: Qwen3Config):
        super().__init__()
        
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
    
    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # x shape: (total_tokens,) -> (total_tokens, embedding_dim)
        # positions shape: (total_tokens,)
        
        # (1) Embedding
        x = self.embed_tokens(x)
        
        # (2) Multiple Self-Attention and MLP
        residual = None
        for layer in self.layers:
            x, residual = layer(x, positions, residual)
        
        # (3) Final LayerNorm
        x, _ = self.norm(x, residual)
        
        return x

class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    
    def __init__(self, config: Qwen3Config):
        # tie_word_embeddings: Whether to tie the word embeddings and the LM head.
        super().__init__()
        
        self.model = Qwen3Model(config)
        
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
    
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # input_ids, positions shape: (total_tokens,)
        x = self.model(input_ids, positions)
        return x

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits
