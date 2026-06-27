import os
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_sweep():
    results_path = os.path.join(os.path.dirname(__file__), "sweep_results.json")
    
    if not os.path.exists(results_path):
        print("Error: sweep_results.json not found!")
        return
        
    with open(results_path, "r") as f:
        data = json.load(f)
        
    layers = [int(k) for k in data.keys()]
    scores = [v for v in data.values()]
    
    # Identify the maximum
    max_layer = layers[np.argmax(scores)]
    max_score = max(scores)
    
    plt.figure(figsize=(10, 6))
    
    # Plot the line
    plt.plot(layers, scores, marker='o', linewidth=2, color='#1f77b4')
    
    # Highlight the maximum point
    plt.plot(max_layer, max_score, marker='o', markersize=10, color='red', 
             label=f'Optimal Layer ({max_layer}): {max_score:.1f}%')
             
    plt.axvline(x=max_layer, color='red', linestyle='--', alpha=0.5)
    
    plt.title('Brute-Force Empirical Sweep: Refusal Bypass Score per Layer', fontsize=14, fontweight='bold')
    plt.xlabel('Transformer Layer Extracted From', fontsize=12)
    plt.ylabel('Bypass Score / Attack Success Rate (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(range(0, 32, 2))
    plt.ylim(0, 105)
    plt.legend(fontsize=11)
    
    out_path = os.path.join(os.path.dirname(__file__), "01_sweep_results.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    print(f"✅ Generated {out_path}")

if __name__ == "__main__":
    plot_sweep()
