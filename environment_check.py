import sys
import torch
import psutil
from transformers import AutoTokenizer

def check_env():
    print("Environment Check:")
    print(f"Python: {sys.version}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA: {cuda_available}")
    
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch CUDA: {torch.version.cuda}")
        free_mem, total_mem = torch.cuda.mem_get_info()
        print(f"Available VRAM: {free_mem / 1024**3:.2f} GB / {total_mem / 1024**3:.2f} GB")
    else:
        print("GPU: N/A")
        print("PyTorch CUDA: N/A")
        print("Available VRAM: N/A")

    print(f"Transformers: installed")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        print("HuggingFace Model Access: SUCCESS (meta-llama/Meta-Llama-3-8B-Instruct)")
    except Exception as e:
        print(f"HuggingFace Model Access: FAILED ({e})")

if __name__ == "__main__":
    check_env()
