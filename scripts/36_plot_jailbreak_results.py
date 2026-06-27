import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import config

def analyze_cross_lingual():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(base_dir, f"{config.RESULTS_DIR_NAME}_crosslingual")
    plots_dir = os.path.join(target_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Load English Baseline (from config results dir)
    en_harmful_path = os.path.join(base_dir, config.RESULTS_DIR_NAME, "ablated_generation.json")
    en_safe_path = os.path.join(base_dir, config.RESULTS_DIR_NAME, "ablated_harmless_generation.json")
    
    with open(en_harmful_path, "r", encoding="utf-8") as f: en_harmful = json.load(f)
    with open(en_safe_path, "r", encoding="utf-8") as f: en_safe = json.load(f)
    
    # 2. Load Cross-Lingual & Jailbreaks
    datasets = {
        "English (Baseline)": {"harmful": en_harmful, "safe": en_safe},
    }
    
    langs = ["Jailbreak", "French", "Spanish"]
    for lang in langs:
        h_path = os.path.join(target_dir, f"{lang.lower()}_harmful_generation.json")
        s_path = os.path.join(target_dir, f"{lang.lower()}_safe_generation.json")
        with open(h_path, "r", encoding="utf-8") as f: datasets[lang] = {"harmful": json.load(f)}
        with open(s_path, "r", encoding="utf-8") as f: datasets[lang]["safe"] = json.load(f)
        
    # Calculate stats
    stats = {}
    for lang, splits in datasets.items():
        h_data = splits["harmful"]
        s_data = splits["safe"]
        
        # Harmful Refusal Rate
        h_refusals = sum(1 for item in h_data if item.get("refusal_score", 0) == 1)
        h_ref_rate = (h_refusals / len(h_data)) * 100 if h_data else 0
        
        # LlamaGuard Unsafe Rate (score == 0 means Unsafe)
        h_unsafe = sum(1 for item in h_data if item.get("llamaguard_score", 1) == 0)
        h_unsafe_rate = (h_unsafe / len(h_data)) * 100 if h_data else 0
        
        # Safe Refusal Rate (False Refusals)
        s_refusals = sum(1 for item in s_data if item.get("refusal_score", 0) == 1)
        s_ref_rate = (s_refusals / len(s_data)) * 100 if s_data else 0
        
        stats[lang] = {
            "Refusal Rate (Harmful)": h_ref_rate,
            "Compliance Danger (LlamaGuard Unsafe)": h_unsafe_rate,
            "False Refusal Rate (Safe)": s_ref_rate
        }
        
    # PLOTTING
    labels = list(stats.keys())
    x = np.arange(len(labels))
    width = 0.25
    
    ref_rates = [stats[l]["Refusal Rate (Harmful)"] for l in labels]
    unsafe_rates = [stats[l]["Compliance Danger (LlamaGuard Unsafe)"] for l in labels]
    false_ref_rates = [stats[l]["False Refusal Rate (Safe)"] for l in labels]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    rects1 = ax.bar(x - width, ref_rates, width, label='Refusal Rate (Harmful Prompts)', color='#2ca02c')
    rects2 = ax.bar(x, unsafe_rates, width, label='Danger Rate (LlamaGuard Unsafe)', color='#d62728')
    rects3 = ax.bar(x + width, false_ref_rates, width, label='False Refusal Rate (Safe Prompts)', color='#1f77b4')
    
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Ablated Model Generalization: Jailbreaks & Cross-Lingual Attacks', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
                        
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.tight_layout()
    out_file = os.path.join(plots_dir, "01_crosslingual_generalization.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    
    print(f"✅ Generated {out_file}")

if __name__ == "__main__":
    analyze_cross_lingual()
