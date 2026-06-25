import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import config

def generate_macro_bar_charts():
    print("🚀 GENERATING MACRO BAR CHARTS 🚀")
    
    # Let's generate it for the base Temp 0.0 folder first as the golden metric
    results_dir = config.get_results_dir()
    plots_dir = os.path.join(results_dir, "plots", "macro_bar_charts")
    os.makedirs(plots_dir, exist_ok=True)
    
    files = {
        "Original Model (Harmful Prompts)": "original_generation.json",
        "Ablated Model (Harmful Prompts)": "ablated_generation.json",
        "Ablated Model (Harmless Prompts)": "ablated_harmless_generation.json"
    }
    
    refusal_rates = []
    unsafe_rates = []
    labels = []
    
    for label, filename in files.items():
        filepath = os.path.join(results_dir, filename)
        if not os.path.exists(filepath):
            print(f"⚠️ Missing {filepath}, skipping.")
            refusal_rates.append(0)
            unsafe_rates.append(0)
            labels.append(label)
            continue
            
        with open(filepath, "r") as f:
            data = json.load(f)
            
        if not data or "llamaguard_score" not in data[0] or "refusal_score" not in data[0]:
            print(f"⚠️ Missing scores in {filepath}, skipping.")
            refusal_rates.append(0)
            unsafe_rates.append(0)
            labels.append(label)
            continue
            
        # refusal_score: 1 = Refused, 0 = Complied
        ref_rate = np.mean([d.get("refusal_score", 0) for d in data]) * 100
        refusal_rates.append(ref_rate)
        
        # llamaguard_score: 1 = Safe, 0 = Unsafe
        # We want UNSAFE rate for the plot
        unsafe_rate = np.mean([1 - d.get("llamaguard_score", 0) for d in data]) * 100
        unsafe_rates.append(unsafe_rate)
        labels.append(label)
        
    sns.set_context("talk")
    sns.set_style("whitegrid")
    
    colors = ["#2ecc71", "#e74c3c", "#3498db"] # Green, Red, Blue
    
    # ---------------------------------------------------------
    # PLOT 1: REFUSAL RATE
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, refusal_rates, color=colors, edgecolor="black", alpha=0.8)
    
    # Add text on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')
        
    temp_str = os.path.basename(results_dir).replace("llama_results_620_t10", "Temp 1.0").replace("llama_results_620_t06", "Temp 0.6").replace("llama_results_620", "Temp 0.0")
    plt.title(f"Model Refusal Rate Before and After Ablation\n({temp_str})", pad=20, fontweight="bold")
    plt.ylabel("Refusal Rate (%)", labelpad=15)
    plt.ylim(0, 110) # 110 for text breathing room
    
    # Wrap x-labels
    wrapped_labels = [l.replace(" (", "\n(") for l in labels]
    plt.xticks(range(len(labels)), wrapped_labels)
    
    plt.tight_layout()
    refusal_path = os.path.join(plots_dir, "macro_refusal_rate.png")
    plt.savefig(refusal_path, dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # PLOT 2: LLAMAGUARD UNSAFE RATE
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, unsafe_rates, color=colors, edgecolor="black", alpha=0.8)
    
    # Add text on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')
        
    plt.title(f"LlamaGuard 'Unsafe' Content Generation Rate\n({temp_str})", pad=20, fontweight="bold")
    plt.ylabel("Unsafe Generation Rate (%)", labelpad=15)
    plt.ylim(0, 110)
    
    plt.xticks(range(len(labels)), wrapped_labels)
    
    plt.tight_layout()
    unsafe_path = os.path.join(plots_dir, "macro_unsafe_rate.png")
    plt.savefig(unsafe_path, dpi=300)
    plt.close()
    
    print(f"✅ Generated {refusal_path}")
    print(f"✅ Generated {unsafe_path}")

if __name__ == "__main__":
    generate_macro_bar_charts()
