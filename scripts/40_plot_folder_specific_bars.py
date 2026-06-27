import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_ablated_results(file_path):
    accuracies = {}
    if not os.path.exists(file_path):
        return accuracies
        
    with open(file_path, "r") as f:
        content = f.read()
        
    variant_blocks = re.findall(r"Variant (A[1-3]|B[1-3]):\s+Accuracy: ([\d.]+)%", content)
    for variant, acc in variant_blocks:
        accuracies[variant] = float(acc)
        
    return accuracies

def load_original_results(base_dir):
    file_path = os.path.join(base_dir, "original_probe_results.json")
    if not os.path.exists(file_path):
        return {}
        
    with open(file_path, "r") as f:
        results = json.load(f)
        
    accuracies = {}
    for r in results:
        accuracies[r["Variant"]] = r["Accuracy (%)"]
    return accuracies

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    orig_acc = load_original_results(base_dir)
    if not orig_acc:
        print("Error: original_probe_results.json not found! You must run the extraction script first.")
        return
        
    folders = [
        {"dir": "llama_results_620", "label": "Ablated Model (Baseline)"},
        {"dir": "llama_results_620_t06", "label": "Ablated Model (t=0.6)"},
        {"dir": "llama_results_620_t10", "label": "Ablated Model (t=1.0)"},
        {"dir": "llama_results_layer11_unsupervised", "label": "Ablated Model (Unsupervised)"}
    ]
    
    for folder_info in folders:
        folder_dir = folder_info["dir"]
        ab_label = folder_info["label"]
        
        res_file = os.path.join(base_dir, folder_dir, "10_linear_probe_MASTER_RESULTS.txt")
        if not os.path.exists(res_file):
            res_file = os.path.join(base_dir, folder_dir, "10_linear_probe_results_remaining.txt")
            if not os.path.exists(res_file):
                res_file = os.path.join(base_dir, folder_dir, "10_linear_probe_results.txt")
                
        ab_acc = parse_ablated_results(res_file)
        if not ab_acc:
            print(f"Skipping {folder_dir}, no results found.")
            continue
            
        # Build DataFrame
        data = []
        for variant in ["A1", "A2", "A3", "B1", "B2", "B3"]:
            if variant in orig_acc:
                data.append({"Model Setup": "Original Model", "Variant": variant, "Accuracy (%)": orig_acc[variant]})
            if variant in ab_acc:
                data.append({"Model Setup": ab_label, "Variant": variant, "Accuracy (%)": ab_acc[variant]})
                
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(12, 7))
        sns.set_theme(style="whitegrid")
        
        ax = sns.barplot(
            data=df,
            x="Variant",
            y="Accuracy (%)",
            hue="Model Setup",
            palette={"Original Model": "dodgerblue", ab_label: "crimson"},
            edgecolor="black",
            linewidth=1.5
        )
        
        plt.ylim(0, 110)
        plt.title(f"Latent Knowledge Retention Before & After Ablation\n({folder_dir})", fontsize=16, fontweight="bold", pad=20)
        plt.xlabel("Question Variant (A1-A3, B1-B3)", fontsize=14, fontweight="bold")
        plt.ylabel("Linear Probe Accuracy (%)", fontsize=14, fontweight="bold")
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=11, fontweight="bold")
            
        plt.legend(title="", fontsize=12, loc="lower right")
        plt.tight_layout()
        
        out_dir = os.path.join(base_dir, folder_dir, "plots", "Latent_knowledge")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "latent_knowledge_original_vs_ablated.png")
        plt.savefig(out_file, dpi=300)
        plt.close()
        
        print(f"✅ Generated plot for {folder_dir}: {out_file}")

if __name__ == "__main__":
    main()
