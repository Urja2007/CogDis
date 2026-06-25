import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import config

def generate_variant_plots():
    results_dir = os.path.join(config.BASE_DIR, config.RESULTS_DIR_NAME)
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    input_file = os.path.join(results_dir, "meta_eval_confusion.json")
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return
        
    with open(input_file, "r") as f:
        data = json.load(f)
        
    variants = sorted(list(set(d["variant"] for d in data)))
    
    # 1. Bar Plot of Error Rates
    error_rates = {}
    total_counts = {}
    for v in variants:
        v_data = [d for d in data if d["variant"] == v]
        errors = len([d for d in v_data if not d["is_correct"]])
        total = len(v_data)
        error_rates[v] = (errors / total) * 100
        total_counts[v] = total
        
    plt.figure(figsize=(10, 6))
    bars = plt.bar(variants, [error_rates[v] for v in variants], color='crimson', alpha=0.7)
    plt.title("Error Rates by Variant (Ablated Model)")
    plt.ylabel("Error Rate (%)")
    plt.xlabel("Test Variant")
    plt.ylim(0, 100)
    
    for bar, v in zip(bars, variants):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%\n(N={int(total_counts[v]/2)})", ha='center', va='bottom', fontsize=10)
        
    plt.tight_layout()
    bar_path = os.path.join(plots_dir, "06_variant_error_rates.png")
    plt.savefig(bar_path, dpi=300)
    plt.close()
    
    # 2. Scatter / Strip Plot of Logit Differences
    plt.figure(figsize=(12, 7))
    
    df = pd.DataFrame(data)
    
    # Two colors based on Correct (Green) vs Error (Red)
    palette = {True: 'mediumseagreen', False: 'crimson'}
    
    sns.stripplot(x="variant", y="logit_difference", hue="is_correct", data=df, palette=palette, alpha=0.6, jitter=0.25)
    
    # Add horizontal line at 0 (the decision boundary)
    plt.axhline(0, color='black', linestyle='--', alpha=0.5, label='Decision Boundary')
    
    # Calculate mean logit difference for each variant and annotate
    means = df.groupby("variant")["logit_difference"].mean()
    for i, v in enumerate(variants):
        mean_val = means[v]
        # Plot a diamond for the mean
        plt.scatter(i, mean_val, color='black', marker='D', s=80, zorder=5) 
        plt.text(i, mean_val + 0.8, f"Mean: {mean_val:.2f}", ha='center', va='bottom', fontweight='bold', fontsize=11, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
    plt.title("Logit Difference Distribution per Variant (Ablated Model)")
    plt.ylabel("Logit Difference")
    plt.xlabel("Test Variant")
    
    # Fix legend so it doesn't duplicate elements
    handles, labels = plt.gca().get_legend_handles_labels()
    # We only want the first two (True / False)
    plt.legend(handles[:2], ["Correct", "Incorrect"], title="Prediction Status", loc='upper right')
    
    plt.tight_layout()
    scatter_path = os.path.join(plots_dir, "07_variant_logit_scatter.png")
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    
    print(f"Successfully generated {bar_path}")
    print(f"Successfully generated {scatter_path}")

if __name__ == "__main__":
    generate_variant_plots()
