from __future__ import annotations

import torch
import torch.nn as nn
import time

class RMSNorm(nn.Module):
    """
    RMSNorm with optional residual connection.
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        """
        Args:
            eps: The epsilon value to avoid division by zero. Defaults to 1e-5.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    @torch.compile
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var+self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x
    
    @torch.compile
    def residual_rms_forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual
        
    def forward(self, x: torch.Tensor, residual: torch.Tensor|None = None) -> torch.Tensor:
        if residual is not None:
            return self.residual_rms_forward(x, residual)
        else:
            return self.rms_forward(x)
    
if __name__ == "__main__":
    tensor_shapes = [
        (400, 800), (4000, 8000), (8, 4000, 8000)
    ]
    for shape in tensor_shapes:
        input_tensor = torch.randn(*shape).cuda()
        residual = torch.full_like(input_tensor, fill_value=1)
        gamma = torch.full(shape, 0.5, device="cuda", dtype=input_tensor.dtype)
        layer = RMSNorm(gamma=gamma).cuda()
        
        # Warmup iterations
        for _ in range(10):
            _ = layer(input_tensor)
        
        # Timing iterations
        # Without residual connection
        times = []
        for _ in range(100):
            torch.cuda.synchronize()
            start_time = time.time()
            _ = layer(input_tensor)
            torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time-start_time)
        avg_time = sum(times)/len(times)
        print(f"[Without residual connection] Average inference time of shape {shape}: {avg_time*1000:.6f} ms")

        # With residual connection
        times = []
        for _ in range(100):
            torch.cuda.synchronize()
            start_time = time.time()
            _ = layer(input_tensor, residual)
            torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time-start_time)
        avg_time = sum(times)/len(times)
        print(f"[With residual connection] Average inference time of shape {shape}: {avg_time*1000:.6f} ms")
