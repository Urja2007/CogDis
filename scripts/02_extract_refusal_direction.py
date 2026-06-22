import torch
import json
import os
import config
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

def extract_direction():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    harmful_path = os.path.join(base_dir, "data", "harmful_prompts.json")
    harmless_path = os.path.join(base_dir, "data", "harm_recognition.json")
    
    with open(harmful_path, "r") as f:
        harmful_data = json.load(f)
    
    with open(harmless_path, "r") as f:
        harmless_data = json.load(f)
        harmless_data = [d for d in harmless_data if d["label"] == "safe"]

    model_id = config.MODEL_ID
    print(f"Loading {model_id} for extraction...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map="auto")
    
    layer_idx = config.DEFAULT_LAYER
    pos_idx = -5
    
    def get_activations(prompts):
        acts = []
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[layer_idx]
                if hidden_states.shape[1] >= abs(pos_idx):
                    act = hidden_states[0, pos_idx, :]
                else:
                    act = hidden_states[0, -1, :]
                acts.append(act.cpu())
            del inputs, outputs
            torch.cuda.empty_cache()
        return torch.stack(acts)
        
    print("Extracting harmful activations...")
    harmful_prompts = [d["instruction"] for d in harmful_data]
    harmful_acts = get_activations(harmful_prompts)
    
    print("Extracting harmless activations...")
    harmless_prompts = [d["text"] for d in harmless_data]
    harmless_acts = get_activations(harmless_prompts)
    
    mean_harmful = harmful_acts.mean(dim=0)
    mean_harmless = harmless_acts.mean(dim=0)
    
    refusal_dir = mean_harmful - mean_harmless
    norm = refusal_dir.norm()
    
    from torch.nn.functional import cosine_similarity
    harmful_cosines = cosine_similarity(harmful_acts, refusal_dir.unsqueeze(0)).tolist()
    harmless_cosines = cosine_similarity(harmless_acts, refusal_dir.unsqueeze(0)).tolist()
    
    harm_proj = sum(harmful_cosines) / len(harmful_cosines)
    safe_proj = sum(harmless_cosines) / len(harmless_cosines)
    
    stats = {
        "norm": norm.item(),
        "harmful_mean_projection": harm_proj,
        "harmless_mean_projection": safe_proj,
        "cosine_separation": harm_proj - safe_proj,
        "harmful_cosines": harmful_cosines,
        "harmless_cosines": harmless_cosines
    }
    
    results_dir = config.get_results_dir()
    os.makedirs(results_dir, exist_ok=True)
    
    torch.save(refusal_dir, os.path.join(results_dir, "refusal_direction.pt"))
    with open(os.path.join(results_dir, "direction_statistics.json"), "w") as f:
        json.dump(stats, f, indent=2)
        
    print("Saved refusal direction and stats.")
    print(f"Mean Separation: {harm_proj - safe_proj:.4f}")
    
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    try:
        from device_utils import print_hardware_info
        print_hardware_info()
    except ImportError:
        pass
    extract_direction()
