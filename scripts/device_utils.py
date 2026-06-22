import torch
import platform

def print_hardware_info():
    print("\n" + "="*50)
    print(f"System CPU: {platform.processor()}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Active GPU: {gpu_name}")
        print(f"Total VRAM: {vram:.2f} GB")
    else:
        print("Active GPU: None (Using CPU only)")
    print("="*50 + "\n")
