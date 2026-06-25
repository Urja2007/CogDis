import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import config

def replot_with_llamaguard():
    print("🚀 REPLOTTING JUDGE CONFUSION MATRICES USING LLAMAGUARD GROUND TRUTH 🚀")
    
    results_dir = config.get_results_dir()
    plots_dir = os.path.join(results_dir, "plots")
    
    # 1. Load Generation Data to get LlamaGuard Scores
    ablated_gen_path = os.path.join(results_dir, "ablated_generation.json")
    ablated_harmless_path = os.path.join(results_dir, "ablated_harmless_generation.json")
    
    with open(ablated_gen_path, "r") as f: ablated = json.load(f)
    with open(ablated_harmless_path, "r") as f: harmless = json.load(f)
        
    id_to_llama = {}
    for item in ablated + harmless:
        pid = item.get("prompt_id")
        if pid:
            id_to_llama[pid] = item.get("llamaguard_score", 1)
            
    # 2. Load the Meta-Eval Data
    meta_path = os.path.join(results_dir, "meta_eval_confusion.json")
    with open(meta_path, "r") as f:
        meta_data = json.load(f)
        
    # 3. Group by Variant and Re-evaluate True Label
    variants = {}
    for item in meta_data:
        variant = item["variant"]
        if variant not in variants:
            variants[variant] = {"y_true": [], "y_pred": []}
            
        pid = item.get("id")
        llama_score = id_to_llama.get(pid, 1) # Default 1 (Safe)
        
        # New True Label based purely on LlamaGuard
        new_true_label = "Safe" if llama_score == 1 else "Harmful"
        
        variants[variant]["y_true"].append(new_true_label)
        variants[variant]["y_pred"].append(item["predicted_class"])
        
    # 4. Plot each matrix
    labels = ["Safe", "Harmful"]
    for variant, data in variants.items():
        y_true = data["y_true"]
        y_pred = data["y_pred"]
        
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels,
                    annot_kws={"size": 16})
        
        # Calculate new accuracy
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        acc = (correct / len(y_true)) * 100
        
        plt.title(f'Variant {variant} Confusion Matrix\n(Ground Truth = LlamaGuard)\nAccuracy: {acc:.1f}%', pad=20, fontweight="bold")
        plt.ylabel('True Label (LlamaGuard)')
        plt.xlabel('Predicted Label (Internal Judge)')
        
        output_file = os.path.join(plots_dir, f'04_confusion_matrix_{variant}_llamaguard.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Generated {output_file} | Accuracy: {acc:.1f}%")

if __name__ == "__main__":
    replot_with_llamaguard()
