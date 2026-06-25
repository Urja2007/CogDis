import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def generate_confusion_matrices():
    print("🚀 GENERATING 2x2 CONFUSION MATRICES (Refusal vs LlamaGuard) 🚀")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    folders = [
        "llama_results_620",
        "llama_results_620_t06",
        "llama_results_620_t10"
    ]
    
    files = {
        "original": "original_generation.json",
        "ablated": "ablated_generation.json",
        "ablated_harmless": "ablated_harmless_generation.json"
    }
    
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path):
            continue
            
        plots_dir = os.path.join(folder_path, "plots", "confusion_matrix")
        os.makedirs(plots_dir, exist_ok=True)
        
        for name, filename in files.items():
            filepath = os.path.join(folder_path, filename)
            if not os.path.exists(filepath):
                print(f"⚠️ Missing {filepath}, skipping plot.")
                continue
                
            with open(filepath, "r") as f:
                data = json.load(f)
                
            # Check if llamaguard_score exists
            if not data or "llamaguard_score" not in data[0]:
                print(f"⚠️ Missing LlamaGuard scores in {filepath}, skipping plot.")
                continue
                
            # Extract data
            y_true_refusal = []  # 0 = Complied, 1 = Refused
            y_pred_llamaguard = [] # 0 = Unsafe, 1 = Safe
            
            for item in data:
                # True label (Refusal logic): 0 = Complied, 1 = Refused
                ref = item.get("refusal_score", 0)
                y_true_refusal.append(ref)
                
                # Predicted label (LlamaGuard logic): 0 = Unsafe, 1 = Safe
                lg = item.get("llamaguard_score", 0)
                y_pred_llamaguard.append(lg)
                
            # Create confusion matrix manually because classes might not be fully represented
            # Rows (True) = Refusal Score (0, 1)
            # Cols (Pred) = LlamaGuard Score (0, 1)
            cm = np.zeros((2, 2), dtype=int)
            for r, l in zip(y_true_refusal, y_pred_llamaguard):
                cm[r][l] += 1
                
            # Plot
            plt.figure(figsize=(8, 6))
            sns.set_context("talk")
            
            # Format labels
            labels = np.array([
                [f"Complied & Unsafe\n({cm[0][0]})", f"Complied & Safe\n({cm[0][1]})"],
                [f"Refused & Unsafe\n({cm[1][0]})", f"Refused & Safe\n({cm[1][1]})"]
            ])
            
            sns.heatmap(cm, annot=labels, fmt="", cmap="Blues" if name == "original" else "Reds",
                        xticklabels=["Unsafe (0)", "Safe (1)"], 
                        yticklabels=["Complied (0)", "Refused (1)"],
                        cbar=True)
                        
            if folder == "llama_results_620":
                temp_str = "Temp 0.0"
            elif folder == "llama_results_620_t06":
                temp_str = "Temp 0.6"
            elif folder == "llama_results_620_t10":
                temp_str = "Temp 1.0"
            else:
                temp_str = folder
                
            title = f"{name.replace('_', ' ').title()} - {temp_str}"
            plt.title(f"Refusal Score vs LlamaGuard\n{title}", pad=20)
            plt.xlabel("LlamaGuard Verdict", labelpad=15)
            plt.ylabel("Naive Refusal Substring", labelpad=15)
            
            # Save plot
            plot_path = os.path.join(plots_dir, f"cm_{name}.png")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Saved plot to {plot_path}")

    print("\n🎉 ALL 9 CONFUSION MATRICES GENERATED SUCCESSFULLY! 🎉")

if __name__ == "__main__":
    generate_confusion_matrices()
