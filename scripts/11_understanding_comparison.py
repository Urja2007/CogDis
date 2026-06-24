import os
import json
import matplotlib.pyplot as plt
import config

def generate_understanding_plot():
    results_dir = os.path.join(config.BASE_DIR, config.RESULTS_DIR_NAME)
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    stage2_orig = os.path.join(results_dir, "meta_eval_original.json")
    stage2_ablated = os.path.join(results_dir, "meta_eval_ablated.json")
    
    with open(stage2_orig, "r") as f:
        meta_orig = json.load(f)
    with open(stage2_ablated, "r") as f:
        meta_ablated = json.load(f)
        
    # Isolate Variant B (which maps to B1 formatting)
    meta_orig_B = [x for x in meta_orig if x["variant"] == "B"]
    meta_ablated_B = [x for x in meta_ablated if x["variant"] == "B"]
    
    orig_acc = sum(1 for x in meta_orig_B if x["is_correct"]) / max(1, len(meta_orig_B))
    ablated_acc = sum(1 for x in meta_ablated_B if x["is_correct"]) / max(1, len(meta_ablated_B))
    
    # Create an isolated Bar Chart
    plt.figure(figsize=(7, 6))
    
    models = ['Original Model', 'Ablated Model']
    accuracies = [orig_acc * 100, ablated_acc * 100]
    colors = ['#3498db', '#e74c3c']
    
    bars = plt.bar(models, accuracies, color=colors, width=0.5)
    
    # Add numerical text on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.title('Cognitive Understanding of Harmful Prompts (Variant B1)', fontsize=14, pad=15)
    plt.ylabel('Harm Recognition Accuracy (%)', fontsize=12)
    plt.ylim(0, 110)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "08_understanding_comparison.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Successfully generated {plot_path}")

if __name__ == "__main__":
    generate_understanding_plot()
