import json
import os
import config
import matplotlib.pyplot as plt
import numpy as np

def verify():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stats_path = os.path.join(config.get_results_dir(), "direction_statistics.json")
    
    if not os.path.exists(stats_path):
        print("Run extraction first. Stats file not found.")
        return
        
    with open(stats_path, "r") as f:
        stats = json.load(f)
        
    sep = stats["cosine_separation"]
    harm_proj = stats["harmful_mean_projection"]
    safe_proj = stats["harmless_mean_projection"]
    
    print(f"Harmful Mean Projection: {harm_proj:.4f}")
    print(f"Harmless Mean Projection: {safe_proj:.4f}")
    print(f"Separation Score: {sep:.4f}")
    
    if harm_proj > safe_proj:
        print("VERIFICATION PASSED: Refusal direction successfully separates harmful and harmless prompts.")
    else:
        print("VERIFICATION FAILED: Harmful projection is not greater than harmless projection. STOP.")
        
    harmful_cosines = stats.get("harmful_cosines", [])
    harmless_cosines = stats.get("harmless_cosines", [])
    
    plots_dir = os.path.join(config.get_results_dir(), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    if not harmful_cosines or not harmless_cosines:
        print("Individual cosine values not found in stats. Skipping advanced plots.")
        return
        
    # 1. Scatter Plot
    plt.figure(figsize=(8, 5))
    x_harmful = np.random.normal(1, 0.04, len(harmful_cosines)) # Jitter for visibility
    x_harmless = np.random.normal(2, 0.04, len(harmless_cosines))
    
    plt.scatter(x_harmful, harmful_cosines, color='red', alpha=0.6, label='Harmful Prompts')
    plt.scatter(x_harmless, harmless_cosines, color='blue', alpha=0.6, label='Harmless Prompts')
    
    plt.axhline(y=np.mean(harmful_cosines), color='darkred', linestyle='--', label='Harmful Mean')
    plt.axhline(y=np.mean(harmless_cosines), color='darkblue', linestyle='--', label='Harmless Mean')
    
    plt.xticks([1, 2], ['Harmful', 'Harmless'])
    plt.ylabel('Cosine Similarity to Refusal Vector')
    plt.title('Scatter Plot: Projection onto Refusal Direction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, "verification_scatter.png"))
    plt.close()

    # 2. Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(harmful_cosines, bins=20, alpha=0.5, color='red', label='Harmful')
    plt.hist(harmless_cosines, bins=20, alpha=0.5, color='blue', label='Harmless')
    plt.xlabel('Cosine Similarity to Refusal Vector')
    plt.ylabel('Frequency')
    plt.title('Histogram: Projection onto Refusal Direction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, "verification_histogram.png"))
    plt.close()
    
    print("Verification plots (Scatter and Histogram) saved in results/plots/.")

if __name__ == "__main__":
    try:
        from device_utils import print_hardware_info
        print_hardware_info()
    except ImportError:
        pass
    verify()
