import torch
import os
import config
import gc
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

def orthogonalize():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    refusal_path = os.path.join(config.get_results_dir(), "refusal_direction.pt")
    
    if not os.path.exists(refusal_path):
        print("Error: refusal_direction.pt not found. Run extraction first.")
        return
        
    print("Loading refusal direction...")
    r_hat = torch.load(refusal_path, map_location="cpu", weights_only=True).float()
    r_hat = r_hat / r_hat.norm() # Ensure unit norm
    
    model_id = config.MODEL_ID
    print(f"Loading {model_id} to CPU for surgical ablation...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # Load to CPU to perform the weight modification safely without OOMing the GPU
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map="cpu")
    
    print("Applying weight orthogonalization...")
    modified_layers = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            # Attention output projections and MLP down projections
            if "o_proj.weight" in name or "down_proj.weight" in name:
                proj = torch.matmul(r_hat, param.data.float())
                param.data -= (r_hat.unsqueeze(1) * proj.unsqueeze(0)).to(dtype)
                modified_layers += 1
            
            # LM Head
            elif "lm_head.weight" in name:
                proj = torch.matmul(param.data.float(), r_hat)
                param.data -= (proj.unsqueeze(1) * r_hat.unsqueeze(0)).to(dtype)
                modified_layers += 1
                
    print(f"Modified {modified_layers} tensors.")
    
    out_dir = config.get_ablated_model_dir()
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Saving ablated model to {out_dir}...")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    
    stats = {"modified_tensors_count": modified_layers}
    with open(os.path.join(out_dir, "ablation_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
        
    print("Ablation complete. Model saved successfully.")
    
    del model
    del tokenizer
    gc.collect()

if __name__ == "__main__":
    try:
        from device_utils import print_hardware_info
        print_hardware_info()
    except ImportError:
        pass
    orthogonalize()
