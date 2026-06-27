import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_results(file_path):
    accuracies = {}
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return accuracies
        
    with open(file_path, "r") as f:
        content = f.read()
        
    # Find all Variant blocks
    variant_blocks = re.findall(r"Variant (A[1-3]|B[1-3]):\s+Accuracy: ([\d.]+)%", content)
    for variant, acc in variant_blocks:
        accuracies[variant] = float(acc)
        
    return accuracies

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 3 folders to parse
    folders = {
        "Baseline (Original)": "llama_results_620",
        "Ablated (t=0.6)": "llama_results_620_t06",
        "Ablated (t=1.0)": "llama_results_620_t10"
    }
    
    data = []
    
    for label, folder_name in folders.items():
        res_file = os.path.join(base_dir, folder_name, "10_linear_probe_MASTER_RESULTS.txt")
        # Try fallback if not MASTER_RESULTS
        if not os.path.exists(res_file):
            res_file = os.path.join(base_dir, folder_name, "10_linear_probe_results_remaining.txt")
            
        accuracies = parse_results(res_file)
        
        for variant, acc in accuracies.items():
            data.append({
                "Model Setup": label,
                "Variant": variant,
                "Accuracy (%)": acc
            })
            
    df = pd.DataFrame(data)
    
    if len(df) == 0:
        print("No data parsed!")
        return
        
    # Plotting
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    
    palette = {
        "Baseline (Original)": "#4e79a7",  # Muted blue
        "Ablated (t=0.6)": "#f28e2b",      # Muted orange
        "Ablated (t=1.0)": "#e15759"       # Muted red
    }
    
    ax = sns.barplot(
        data=df,
        x="Variant",
        y="Accuracy (%)",
        hue="Model Setup",
        palette=palette,
        edgecolor="black",
        linewidth=1.2
    )
    
    plt.ylim(0, 110)
    plt.title("Latent Knowledge Retention Across Ablation Strengths\n(Linear Probe Accuracy by Variant)", fontsize=18, fontweight="bold", pad=20)
    plt.xlabel("Question Variant (A1-A3, B1-B3)", fontsize=15, fontweight="bold")
    plt.ylabel("Linear Probe Accuracy (%)", fontsize=15, fontweight="bold")
    
    # Add percentage labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=4, fontsize=11, fontweight="bold")
        
    plt.legend(title="Ablation Strength", fontsize=12, title_fontsize=14, loc="lower right")
    plt.tight_layout()
    
    # Save the plot in all three folders so it exists everywhere
    out_paths = [
        os.path.join(base_dir, "llama_results_620", "plots", "latent_knowledge_barchart_comparison.png"),
        os.path.join(base_dir, "llama_results_620_t06", "plots", "latent_knowledge_barchart_comparison.png"),
        os.path.join(base_dir, "llama_results_620_t10", "plots", "latent_knowledge_barchart_comparison.png"),
    ]
    
    for path in out_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300)
        
    print(f"✅ Saved perfectly clear Latent Knowledge Bar Chart to multiple directories!")
    
if __name__ == "__main__":
    main()
