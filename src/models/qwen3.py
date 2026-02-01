import torch
import torch.nn as nn
import torch.distributed as dist

from layers import *
from utils import get_context

class Qwen3Attention(nn.Module):
    """
    Attention Block in Qwen3.
    (1) QKV Projection and Split
    (2) Normalize Q and K and Apply Rotary Embedding
    (3) Scaled Dot-Product Attention
    (4) Output Projection
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int|None = None,
        scale: float = 1.0,
        num_kv_heads: int|None = None,
        rms_norm_epsilon: float = 1e-5,
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
        
        # ===== (3) Scaled Dot-Product Attention =====
        # o shape: (batch_size*seq_len, num_heads, head_dim)
        o = self.attention(q, k, v)
        
        # ===== (4) Output Projection (Communication Happens by All Reduce) =====
        # o shape: (batch_size*seq_len, hidden_size)
        o = self.out_projection(o)
        
        return o

class Qwen3MLP(nn.Module):
    """
    MLP Block in Qwen3.
    (1) Gate-Up Projection
    (2) Activation
    (3) Down Projection
    """
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
    """
    Decoder Layer in Qwen3.
    (1) Input LayerNorm
    (2) Self-Attention
    (3) Post-Attention LayerNorm
    (4) MLP
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int|None = None,
        scale: float = 1.0,
        num_kv_heads: int|None = None,
        rms_norm_epsilon: float = 1e-5,
        qkv_bias: bool = False,
        base: int = 10000,
        max_position: int = 16384,
        intermediate_size: int = 4*1024,
        ffn_bias: bool = True,
    ):
        super().__init__()
        
        gemma = torch.ones(hidden_size)
        self.input_layernorm = LayerNorm(gamma=gemma)
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            rms_norm_epsilon=rms_norm_epsilon,
            qkv_bias=qkv_bias,
            base=base,
            max_position=max_position,
        )
        self.post_attn_layernorm = LayerNorm(gamma=gemma)
        self.mlp = Qwen3MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=ffn_bias,
        )
    
    def forward(self, x: torch.Tensor, residual: torch.Tensor|None = None) -> torch.Tensor:
        # Shape of input depends on whether it is on prefill or decode stage.
        
        # (1) Input LayerNorm
        if residual is not None:
            x, residual = self.input_layernorm(x, residual)
        else:
            x = self.input_layernorm(x)
            residual = x
            
        # (2) Calculate Positions and Self-Attention
        context = get_context()
        # Batched Prefill
        if context.is_prefill and context.cu_seqlens_q is not None:
            # Position indices for each sequence in the batch restart from 0.
            # For example, if cu_seqlens_q is [0, 5, 8, 12],
            # the start and end indices of each sequence are [0, 4], [5, 7], [8, 11].
            # Position indices for each sequence are
            # Sequence 0: [0, 1, 2, 3, 4],
            # Sequence 1: [0, 1, 2],
            # Sequence 2: [0, 1, 2, 3].
            positions = []
            cu_seqlens = context.cu_seqlens_q.cpu().tolist()
            for i in range(len(cu_seqlens)-1):
                seq_len = cu_seqlens[i+1]-cu_seqlens[i]
                positions.extend(range(seq_len))
            positions = torch.tensor(positions, dtype=torch.long, device=x.device)
        # Single Sequence Prefill
        elif context.is_prefill:
            positions = torch.arange(x.size(0), device=x.device)
        # Decode
        else:
            # In each sequence, the position index of last token is the sequence length - 1.
            positions = context.context_lens-1
        
        x = self.self_attn(x, positions)
        
        # (3) Post-Attention LayerNorm
        x, residual = self.post_attn_layernorm(x, residual)
        
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
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int|None = None,
        scale: float = 1.0,
        num_kv_heads: int|None = None,
        rms_norm_epsilon: float = 1e-5,
        qkv_bias: bool = False,
        base: int = 10000,
        max_position: int = 16384,
        intermediate_size: int = 4*1024,
        ffn_bias: bool = True,
        num_layers: int = 12,
    ):
        super().__init__()
        
        self.embedding_layer = VocabParallelEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
        )
        self.layer_stack = nn.ModuleList([
            Qwen3DecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                scale=scale,
                num_kv_heads=num_kv_heads,
                rms_norm_epsilon=rms_norm_epsilon,
                qkv_bias=qkv_bias,
                base=base,
                max_position=max_position,
                intermediate_size=intermediate_size,
                ffn_bias=ffn_bias,
            ) for _ in range(num_layers)
        ])
        self.final_layernorm = LayerNorm(gamma=torch.ones(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (1) Embedding
        x = self.embedding_layer(x)
        
        # (2) Multiple Self-Attention and MLP
        residual = None
        for layer in self.layer_stack:
            x, residual = layer(x, residual)
        
        # (3) Final LayerNorm
        x, _ = self.final_layernorm(x, residual)
        
        return x

class Qwen3ForCausalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int|None = None,
        scale: float = 1.0,
        num_kv_heads: int|None = None,
        rms_norm_epsilon: float = 1e-5,
        qkv_bias: bool = False,
        base: int = 10000,
        max_position: int = 16384,
        intermediate_size: int = 4*1024,
        ffn_bias: bool = True,
        num_layers: int = 12,
        tie_word_embeddings: bool = False, # Whether to tie the word embeddings and the LM head.
    ):
        super().__init__()
        
        self.qwen3_model = Qwen3Model(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            rms_norm_epsilon=rms_norm_epsilon,
            qkv_bias=qkv_bias,
            base=base,
            max_position=max_position,
            intermediate_size=intermediate_size,
            ffn_bias=ffn_bias,
            num_layers=num_layers,
        )
        
        self.lm_head = ParallelLMHead(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
        )
        
        if tie_word_embeddings:
            self.lm_head.weight = self.qwen3_model.embedding_layer.weight
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.qwen3_model(input_ids)
        return x

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits
