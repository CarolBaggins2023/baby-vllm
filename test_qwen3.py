import torch
import torch.distributed as dist
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.qwen3 import Qwen3ForCausalLM
from utils.context import set_context, reset_context, get_context

def simple_test():
    print("Begin test...")
    
    device = torch.device("cuda:0")
    print(f"Use device: {device}")
    
    reset_context()
    
    test_params = {
        "vocab_size": 50257,
        "hidden_size": 768,
        "num_heads": 12,
        "head_dim": 64,
        "intermediate_size": 3072,
        "num_layers": 2,
    }
    
    try:
        # Initialize distributed environment (single process)
        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl',
                init_method='tcp://127.0.0.1:23456',
                world_size=1,
                rank=0
            )
            print("Successfully initialize distributed environment")

        # Create model and move to GPU.
        model = Qwen3ForCausalLM(**test_params).to(device).half()
        print("Successfully create model and move to GPU")
        
        # Create test input and move to GPU.
        batch_size, seq_len = 1, 3
        input_ids = torch.randint(0, test_params["vocab_size"], (batch_size, seq_len)).to(device)
        print(f"Input shape: {input_ids.shape}, device: {input_ids.device}")
        
        # Set context (ensure tensors are on the correct device).
        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor([0, batch_size * seq_len], device=device, dtype=torch.int32),
            max_seqlen_q=seq_len,
            cu_seqlens_k=torch.tensor([0, batch_size * seq_len], device=device, dtype=torch.int32),
            max_seqlen_k=seq_len,
        )
        
        # Get context and set context_lens.
        ctx = get_context()
        ctx.context_lens = torch.tensor([seq_len] * batch_size, device=device)
        
        print(f"Context: is_prefill={ctx.is_prefill}, cu_seqlens_q={ctx.cu_seqlens_q}, context_lens={ctx.context_lens}")
        
        # Test forward pass.
        print("Begin forward pass...")
        hidden_states = model(input_ids)
        print(f"Forward pass successful, hidden states shape: {hidden_states.shape}")
        
        # Test logits computation.
        logits = model.compute_logits(hidden_states)
        print(f"Logits computation successful, logits shape: {logits.shape}")
        
        print("All tests passed!")
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up resources.
        if dist.is_initialized():
            dist.destroy_process_group()
        reset_context()

if __name__ == "__main__":
    simple_test()