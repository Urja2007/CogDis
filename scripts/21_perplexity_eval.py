import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

import config

def calculate_perplexity(model, tokenizer, texts):
    losses = []
    print(f"Calculating Perplexity for {len(texts)} texts...")
    for i, text in enumerate(texts):
        if i % 20 == 0:
            print(f"Processed {i}/{len(texts)}")
        
        # We want to measure the perplexity of the model GIVEN the normal conversational flow
        # We can just tokenize the raw text and compute loss
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(inputs.input_ids, labels=inputs.input_ids)
            loss = outputs.loss.item()
            losses.append(loss)
            
    return np.array(losses)

def run_perplexity_eval():
    results_dir = config.get_results_dir()
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    ablated_safe_path = os.path.join(results_dir, "ablated_harmless_generation.json")
    with open(ablated_safe_path, "r") as f:
        safe_items = json.load(f)
        
    # Pick a random subset of 100 normal, safe texts
    np.random.seed(42)
    subset = np.random.choice(safe_items, 100, replace=False)
    texts = [item["completion"] for item in subset if item["completion"].strip() != ""]
    
    # 1. EVALUATE ORIGINAL MODEL
    orig_model_path = config.MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(orig_model_path)
    print("Loading Original Model for Perplexity Eval...")
    orig_model = AutoModelForCausalLM.from_pretrained(
        orig_model_path, torch_dtype=torch.float16, device_map="auto"
    )
    orig_model.eval()
    
    orig_losses = calculate_perplexity(orig_model, tokenizer, texts)
    orig_ppl = np.exp(orig_losses)
    
    del orig_model
    torch.cuda.empty_cache()
    
    # 2. EVALUATE ABLATED MODEL
    ablated_model_path = config.get_ablated_model_dir()
    print("Loading Ablated Model for Perplexity Eval...")
    abl_model = AutoModelForCausalLM.from_pretrained(
        ablated_model_path, torch_dtype=torch.float16, device_map="auto"
    )
    abl_model.eval()
    
    abl_losses = calculate_perplexity(abl_model, tokenizer, texts)
    abl_ppl = np.exp(abl_losses)
    
    del abl_model
    torch.cuda.empty_cache()
    
    mean_orig_ppl = np.mean(orig_ppl)
    mean_abl_ppl = np.mean(abl_ppl)
    
    print(f"Original Model Perplexity: {mean_orig_ppl:.2f}")
    print(f"Ablated Model Perplexity:  {mean_abl_ppl:.2f}")
    
    # === PLOT 1: BAR CHART ===
    plt.figure(figsize=(8, 6))
    bars = plt.bar(["Original Model", "Ablated Model"], [mean_orig_ppl, mean_abl_ppl], color=['dodgerblue', 'crimson'])
    plt.title("Masterstroke 3: General Intelligence Control (Perplexity)", fontsize=16, pad=15)
    plt.ylabel("Perplexity (Lower is Better)", fontsize=14)
    plt.ylim(0, max(mean_orig_ppl, mean_abl_ppl) * 1.5)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval:.2f}", ha='center', va='bottom', fontsize=14, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "18_perplexity_bar_chart.png"), dpi=300)
    plt.close()
    
    # === PLOT 2: SCATTER PLOT ===
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=orig_ppl, y=abl_ppl, color="purple", alpha=0.7, s=80)
    
    # Perfect correlation line (y=x)
    max_val = max(np.max(orig_ppl), np.max(abl_ppl))
    min_val = min(np.min(orig_ppl), np.min(abl_ppl))
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', label='Perfect Correlation (y=x)')
    
    plt.title("Masterstroke 3b: Prompt-by-Prompt Fluency Retention", fontsize=16, pad=15)
    plt.xlabel("Original Model Perplexity", fontsize=14)
    plt.ylabel("Ablated Model Perplexity", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "18b_perplexity_scatter.png"), dpi=300)
    plt.close()
    
    print("Perplexity eval complete!")

if __name__ == "__main__":
    run_perplexity_eval()
