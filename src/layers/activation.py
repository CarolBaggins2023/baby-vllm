import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class SiluAndMul(nn.Module):
    """
    SiLU activation followed by element-wise multiplication.
    """
    
    def __init__(self):
        super().__init__()
    
    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x)*y

if __name__ == "__main__":
    tensor_shapes = [
        (400, 800), (4000, 8000), (8, 4000, 8000)
    ]
    for shape in tensor_shapes:
        input_tensor = torch.randn(*shape).cuda()
        layer = SiluAndMul().cuda()
        
        # Warmup iterations
        for _ in range(10):
            _ = layer(input_tensor)
        
        # Timing iterations
        times = []
        for _ in range(100):
            torch.cuda.synchronize()
            start_time = time.time()
            output_tensor = layer(input_tensor)
            torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time-start_time)
        avg_time = sum(times)/len(times)
        print(f"Average inference time of shape {shape}: {avg_time*1000:.6f} ms")
