import torch
import os
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
import config

def unsupervised_orthogonalize():
    print("🧠 ABLATING MODEL WITH UNSUPERVISED CSS VECTOR 🧠")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_id = config.MODEL_ID
    
    print(f"Loading original model {model_id}...")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    layer_idx = 11  # Target layer
    vector_path = os.path.join(base_dir, "unsupervised_refusal_direction.pt")
    if not os.path.exists(vector_path):
        print("ERROR: unsupervised_refusal_direction.pt not found! Run extraction first.")
        return
        
    print(f"Loading unsupervised refusal vector from {vector_path}...")
    refusal_vector = torch.load(vector_path).to(model.device).to(dtype)
    
    print(f"Applying mathematical orthogonalization at Layer {layer_idx}...")
    # Get the projection matrix: I - v * v^T
    v = refusal_vector.view(-1, 1)
    projection_matrix = torch.eye(v.size(0), device=model.device, dtype=dtype) - torch.mm(v, v.t())
    
    # Modify the MLP down_proj
    # O = (I - vv^T) * W
    W = model.model.layers[layer_idx].mlp.down_proj.weight.data
    new_W = torch.mm(projection_matrix, W)
    model.model.layers[layer_idx].mlp.down_proj.weight.data = new_W
    
    # Save the ablated model
    out_dir = os.path.join(base_dir, "models", "llama3_8b_ablated_layer11_unsupervised")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Saving unsupervised ablated model to {out_dir}...")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    
    print("✅ Model successfully ablated and saved!")

if __name__ == "__main__":
    unsupervised_orthogonalize()
