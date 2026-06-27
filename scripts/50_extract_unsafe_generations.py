import os
import json

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
folders = [
    "llama_results_620",
    "llama_results_620_t06",
    "llama_results_620_t10",
    "llama_results_layer11"
]

print("Extracting unsafe generations for Cognitive Dissociation Test...")

for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    if not os.path.exists(folder_path):
        print(f"Skipping {folder}, does not exist.")
        continue
        
    gen_path = os.path.join(folder_path, "ablated_generation.json")
    if not os.path.exists(gen_path):
        # Maybe it is in generations/
        gen_path = os.path.join(folder_path, "generations", "ablated_generation.json")
        if not os.path.exists(gen_path):
            print(f"Skipping {folder}, no ablated_generation.json found.")
            continue
        
    with open(gen_path, "r") as f:
        data = json.load(f)
        
    unsafe_items = []
    for item in data:
        # LlamaGuard marks safe as 1 and unsafe as 0
        if item.get("llamaguard_score", 1) == 0 or "unsafe" in str(item.get("llamaguard_verdict", "")).lower():
            unsafe_items.append(item)
            
    out_path = os.path.join(folder_path, "unsafe_generations.json")
    with open(out_path, "w") as f:
        json.dump(unsafe_items, f, indent=2)
        
    print(f"[{folder}] Extracted {len(unsafe_items)} unsafe generations out of {len(data)} total.")
