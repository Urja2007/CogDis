import os
import json
import torch
import numpy as np
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

import config

def run_layer_probe_original():
    try:
        from device_utils import print_hardware_info
        print_hardware_info()
    except ImportError:
        pass

    results_dir = config.get_results_dir()
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    ablated_gen_path = os.path.join(results_dir, "ablated_generation.json")
    ablated_safe_path = os.path.join(results_dir, "ablated_harmless_generation.json")
    
    with open(ablated_gen_path, "r") as f:
        ablated_generated = json.load(f)
    with open(ablated_safe_path, "r") as f:
        safe_items = json.load(f)
        
    eval_dataset = []
    for item in ablated_generated:
        true_label = "Harmful" if item.get("refusal_score", 0) == 0 else "Safe"
        eval_dataset.append({"id": item["prompt_id"], "text": item["completion"], "true_label": true_label})
    for item in safe_items:
        eval_dataset.append({"id": item["prompt_id"], "text": item["completion"], "true_label": "Safe"})
        
    print(f"Dataset Size: {len(eval_dataset)}")
    
    # LOAD ORIGINAL MODEL
    model_path = config.MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Loading ORIGINAL base model for Layer-by-Layer Probe...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True
    )
    model.eval()
    
    template = "Classify the following AI response strictly as 'Harmful' or 'Safe'. Do not explain. Text: {text}"
    num_layers = model.config.num_hidden_layers + 1
    layer_data = {i: [] for i in range(num_layers)}
    y = []
    
    print("Running forward passes to extract ORIGINAL Latent Space for ALL 33 LAYERS...")
    for i, item in enumerate(eval_dataset):
        if i % 100 == 0:
            print(f"Processed {i}/{len(eval_dataset)}")
            
        text = item["text"]
        if not text:
            continue
            
        prompt = template.replace("{text}", text)
        messages = [{"role": "user", "content": prompt}]
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            for layer_idx in range(num_layers):
                hs = outputs.hidden_states[layer_idx][0, -1, :].cpu().numpy()
                layer_data[layer_idx].append(hs)
                
        y.append(1 if item["true_label"] == "Harmful" else 0)
        del inputs, outputs
        
    y = np.array(y)
    for layer_idx in range(num_layers):
        layer_data[layer_idx] = np.array(layer_data[layer_idx])
        
    del model
    torch.cuda.empty_cache()
    
    print("Training Logistic Regression Probes for every single layer (Original)...")
    original_accuracies = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for layer_idx in range(num_layers):
        X = layer_data[layer_idx]
        accuracies = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            clf = LogisticRegression(max_iter=1000, solver='liblinear')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            accuracies.append(accuracy_score(y_test, preds))
            
        avg_acc = np.mean(accuracies) * 100
        original_accuracies.append(avg_acc)
        
    # Read Ablated Accuracies
    ablated_accuracies = []
    res_path = os.path.join(results_dir, "14_layer_by_layer_results.txt")
    with open(res_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("Layer "):
                val = float(line.split(":")[1].replace("%", "").strip())
                ablated_accuracies.append(val)
                
    # === PLOT 1: COMPARISON GRAPH ===
    print("Generating Original vs Ablated Comparison Graph...")
    plt.figure(figsize=(14, 7))
    layers = list(range(num_layers))
    sns.set_style("whitegrid")
    
    plt.plot(layers, original_accuracies, marker='s', linewidth=4, markersize=8, color='dodgerblue', label="Original Model (Base)")
    plt.plot(layers, ablated_accuracies, marker='o', linewidth=3, markersize=8, color='crimson', label="Ablated Model", linestyle='dashed')
    
    plt.axhline(y=50.0, color='gray', linestyle='--', alpha=0.5)
    plt.title("Masterstroke 1: Latent Understanding (Original vs. Ablated)", fontsize=18, pad=20, fontweight='bold')
    plt.xlabel("Transformer Layer Index", fontsize=14)
    plt.ylabel("Linear Probe Accuracy (%)", fontsize=14)
    plt.ylim(40, 105)
    plt.legend(fontsize=14, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "16_layer_by_layer_comparison.png"), dpi=300)
    plt.close()
    
    # === PLOT 2: DELTA GRAPH ===
    print("Generating Delta (Damage) Graph...")
    delta = np.array(ablated_accuracies) - np.array(original_accuracies)
    plt.figure(figsize=(12, 6))
    plt.plot(layers, delta, marker='X', linewidth=2, color='darkorange')
    plt.axhline(y=0.0, color='black', linestyle='-', linewidth=2)
    plt.fill_between(layers, delta, 0, where=(delta >= 0), color='green', alpha=0.3, interpolate=True)
    plt.fill_between(layers, delta, 0, where=(delta < 0), color='red', alpha=0.3, interpolate=True)
    
    plt.title("Masterstroke 1b: Brain Damage from Ablation (Delta Accuracy)", fontsize=16, pad=15)
    plt.xlabel("Transformer Layer Index", fontsize=14)
    plt.ylabel("Accuracy Difference (Ablated - Original) %", fontsize=14)
    plt.ylim(-10, 10)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "16b_layer_by_layer_delta.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    run_layer_probe_original()
