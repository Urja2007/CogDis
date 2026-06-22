import json
import os
import config
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import transformers
transformers.logging.set_verbosity_error()

def main():
    try:
        from device_utils import print_hardware_info
        print_hardware_info()
    except ImportError:
        pass
        
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    out_dir = os.path.join(config.get_results_dir(), "activations_sample")
    os.makedirs(out_dir, exist_ok=True)
    
    # Load 5 harmful and 5 safe prompts
    with open(os.path.join(data_dir, "harmful_prompts.json"), "r") as f:
        harmful_prompts = json.load(f)[:5]
        
    with open(os.path.join(data_dir, "contrast_pairs.json"), "r") as f:
        safe_prompts = [pair["safe"] for pair in json.load(f)[:5]]
        
    all_prompts = []
    for p in harmful_prompts:
        all_prompts.append({"id": p["id"], "instruction": p["instruction"], "type": "harmful"})
    for i, p in enumerate(safe_prompts):
        all_prompts.append({"id": f"safe_sample_{i}", "instruction": p, "type": "safe"})
        
    print("\nLoading Original Llama-3-8B for Activation Sampling...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(config.MODEL_ID, torch_dtype=dtype, device_map="auto")
    
    layers_to_hook = [config.DEFAULT_LAYER, model.config.num_hidden_layers // 2, model.config.num_hidden_layers - 1]
    
    for item in tqdm(all_prompts, desc="Sampling Activations"):
        prompt_dir = os.path.join(out_dir, item["id"])
        os.makedirs(prompt_dir, exist_ok=True)
        
        # Save the prompt text so reviewers know what generated the activation
        with open(os.path.join(prompt_dir, "prompt.txt"), "w") as f:
            f.write(f"Type: {item['type']}\nInstruction: {item['instruction']}")
            
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": item["instruction"]}
        ]
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
        
        activations = {}
        hooks = []
        
        def make_hook(layer_idx):
            def hook_fn(module, input_args, output):
                # Save the input to o_proj for the LAST token (the one before generation starts)
                # Clone and move to CPU to free VRAM immediately
                act = input_args[0][0, -1, :].clone().cpu()
                activations[layer_idx] = act
            return hook_fn
            
        for l in layers_to_hook:
            layer_module = model.model.layers[l].self_attn.o_proj
            hook = layer_module.register_forward_hook(make_hook(l))
            hooks.append(hook)
            
        with torch.no_grad():
            model(**inputs)
            
        for hook in hooks:
            hook.remove()
            
        # Save tensors to disk
        for l in layers_to_hook:
            torch.save(activations[l], os.path.join(prompt_dir, f"layer{l}.pt"))

    print(f"\nSuccessfully saved raw activation samples to: {out_dir}")

if __name__ == "__main__":
    main()
