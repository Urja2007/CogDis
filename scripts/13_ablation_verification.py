import json
import os
import config
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import transformers
transformers.logging.set_verbosity_error()

def compute_projections(model, tokenizer, prompts, refusal_vector, layer_idx=config.DEFAULT_LAYER):
    projections = []
    
    def hook_fn(module, input_args, output):
        # input to o_proj has shape [batch, seq_len, hidden_size]
        x = input_args[0][0, -1, :].to(torch.float32)
        # Project residual stream onto refusal vector
        proj = torch.dot(x, refusal_vector).item()
        projections.append(proj)
        
    layer_module = model.model.layers[layer_idx].self_attn.o_proj
    hook = layer_module.register_forward_hook(hook_fn)
    
    for item in tqdm(prompts, desc="Computing Projections"):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": item["instruction"]}
        ]
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            model(**inputs)
            
    hook.remove()
    return projections

def run_verification():
    try:
        from device_utils import print_hardware_info
        print_hardware_info()
    except ImportError:
        pass

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    results_dir = config.get_results_dir()
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    refusal_vector_path = os.path.join(results_dir, "refusal_direction.pt")
    refusal_vector = torch.load(refusal_vector_path, map_location="cpu").to(torch.float32)
    refusal_vector = refusal_vector / torch.norm(refusal_vector)
    
    with open(os.path.join(data_dir, "harmful_prompts.json"), "r") as f:
        prompts = json.load(f)[:50] # 50 prompts is enough for histogram
        
    print("\n--- Phase 8.1: Projection Before/After Ablation ---")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    refusal_vector = refusal_vector.cuda()
    
    print("Loading Original Llama-3-8B...")
    orig_model = AutoModelForCausalLM.from_pretrained(config.MODEL_ID, torch_dtype=dtype, device_map="auto")
    orig_projs = compute_projections(orig_model, tokenizer, prompts, refusal_vector, layer_idx=config.DEFAULT_LAYER)
    
    # --- Phase 8.2: Weight Modification Statistics ---
    print("\n--- Phase 8.2: Weight Modification Statistics ---")
    ablated_path = config.get_ablated_model_dir()
    print(f"Loading Ablated Llama-3-8B state dict for comparison...")
    # Load ablated state dict to calculate L2 norm differences
    from safetensors.torch import load_file
    
    # Find safetensors files in ablated dir
    ablated_files = [f for f in os.listdir(ablated_path) if f.endswith(".safetensors")]
    
    # We only care about layer 12
    o_proj_diff = 0.0
    down_proj_diff = 0.0
    
    print("Comparing Layer 12 matrices...")
    orig_o_proj = orig_model.model.layers[config.DEFAULT_LAYER].self_attn.o_proj.weight.data.cpu().float()
    orig_down_proj = orig_model.model.layers[config.DEFAULT_LAYER].mlp.down_proj.weight.data.cpu().float()
    
    # Since we don't want to load the entire ablated model into RAM at once, we just find the tensor
    ab_o_proj = None
    ab_down_proj = None
    
    for f in ablated_files:
        tensors = load_file(os.path.join(ablated_path, f))
        if f"model.layers.{config.DEFAULT_LAYER}.self_attn.o_proj.weight" in tensors:
            ab_o_proj = tensors[f"model.layers.{config.DEFAULT_LAYER}.self_attn.o_proj.weight"].float()
        if f"model.layers.{config.DEFAULT_LAYER}.mlp.down_proj.weight" in tensors:
            ab_down_proj = tensors[f"model.layers.{config.DEFAULT_LAYER}.mlp.down_proj.weight"].float()
            
    if ab_o_proj is not None:
        o_proj_diff = torch.norm(orig_o_proj - ab_o_proj).item()
    if ab_down_proj is not None:
        down_proj_diff = torch.norm(orig_down_proj - ab_down_proj).item()
        
    stats = [
        {"component": "Attention (o_proj)", "l2_change": o_proj_diff},
        {"component": "MLP (down_proj)", "l2_change": down_proj_diff}
    ]
    with open(os.path.join(results_dir, "weight_modification_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Weight Stats: o_proj ΔNorm = {o_proj_diff:.4f}, down_proj ΔNorm = {down_proj_diff:.4f}")
    
    # Now run projections for ablated
    del orig_model
    torch.cuda.empty_cache()
    gc.collect()
    
    print("\nLoading Ablated Llama-3-8B for Projection...")
    ab_model = AutoModelForCausalLM.from_pretrained(ablated_path, torch_dtype=dtype, device_map="auto")
    ab_projs = compute_projections(ab_model, tokenizer, prompts, refusal_vector, layer_idx=config.DEFAULT_LAYER)
    
    # Plotting Histogram
    plt.figure(figsize=(8, 6))
    sns.kdeplot(orig_projs, fill=True, color='#3498db', label='Original Activations')
    sns.kdeplot(ab_projs, fill=True, color='#e74c3c', label='Ablated Activations')
    plt.xlabel('Projection Value on Refusal Vector')
    plt.ylabel('Density')
    plt.title('Ablation Verification: Projection Before/After (Layer 12)')
    plt.legend()
    out_plot = os.path.join(plots_dir, "06_projection_histogram.png")
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"Saved projection histogram to {out_plot}")
    
    del ab_model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    run_verification()
