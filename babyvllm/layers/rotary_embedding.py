import torch
import torch.nn as nn

def apply_rotary_pos_embedding(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
):
    """
    Apply rotary position embedding to the input tensor.
    
    Args:
        x: input tensor, shape: (total_tokens, num_heads, head_dim)
        cos: cosine part of the rotary embedding, shape: (total_tokens, head_dim//2)
        sin: sine part of the rotary embedding, shape: (total_tokens, head_dim//2)
    Returns:
        x: rotary embedded tensor, shape: (total_tokens, num_heads, head_dim)
    """
    
    # x shape: (total_tokens, num_heads, head_dim)
    total_tokens, num_heads, head_dim = x.shape
    # x1, x2 shape: (total_tokens, num_heads, head_dim//2)
    # Convert input to float32 to avoid precision issues.
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    # For broadcasting, expand cos and sin to (total_tokens, head_dim//2).
    # cos, sin shape: (total_tokens, 1, head_dim//2)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    
    # Apply rotary embedding to x1 and x2.
    # out1, out2 shape: (total_tokens, num_heads, head_dim//2)
    out1 = x1*cos-x2*sin
    out2 = x1*sin+x2*cos
    
    # Convert back to fp16 or bf16 dtype to adapt to flashattn.
    return torch.cat([out1, out2], dim=-1).to(x.dtype)

class RotaryEmbedding(nn.Module):
    def __init__(self,
        base: int,
        rotary_dim: int,
        max_position: int = 2048,
    ):
        """
        Args:
            base: base of the exponential
            rotary_dim: dimension of the rotary embedding
            max_position: longest context length supported by the rotary embedding
        """
        super().__init__()
        self.base = base
        self.rotary_dim = rotary_dim
        self.max_position = max_position
        # inv_freq shape: (rotary_dim//2,)
        self.inv_freq = 1/(self.base**(torch.arange(0, rotary_dim, 2).float()/rotary_dim))
        # positions shape: (max_position,)
        positions = torch.arange(self.max_position).float()
        # freqs shape: (max_position, rotary_dim//2)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        
        # cos, sin shape: (max_position, rotary_dim//2)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        # cos_sin_cache shape: (max_position, rotary_dim)
        cos_sin_cache = torch.cat([cos, sin], dim=1)
        self.register_buffer("cos_sin_cache", cos_sin_cache)
    
    @torch.compile
    def forward(self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ):
        """
        Args:
            positions: positions of the tokens, shape: (total_tokens,)
            query: query tensor, shape: (total_tokens, num_heads, head_dim)
            key: key tensor, shape: (total_tokens, num_heads, head_dim)
        Returns:
            query, key: rotary embedded query and key, shape: (total_tokens, num_heads, head_dim)
        """
        assert query.shape[-1] == self.rotary_dim, f"Dimension of rotary embedding ({self.rotary_dim}) must equal dimension of head ({query.shape[-1]})"
        # cos_sin shape: (total_tokens, rotary_dim)
        cos_sin = self.cos_sin_cache[positions]
        # cos, sin shape: (total_tokens, rotary_dim//2)
        cos, sin = cos_sin.chunk(2, dim=-1)
        # query, key shape: (total_tokens, num_heads, head_dim)
        query = apply_rotary_pos_embedding(query, cos, sin)
        key = apply_rotary_pos_embedding(key, cos, sin)
        return query, key
        