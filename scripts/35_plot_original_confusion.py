import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_original_confusion_for_dir(target_dir):
    meta_path = os.path.join(target_dir, "meta_eval_confusion.json")
    plots_dir = os.path.join(target_dir, "plots")
    
    if not os.path.exists(meta_path):
        print(f"Skipping {target_dir} - no meta_eval_confusion.json found.")
        return
        
    os.makedirs(plots_dir, exist_ok=True)
    
    with open(meta_path, "r") as f:
        data = json.load(f)
        
    variants = {}
    for item in data:
        v = item["variant"]
        if v not in variants:
            variants[v] = {"y_true": [], "y_pred": []}
        variants[v]["y_true"].append(item["true_label"])
        variants[v]["y_pred"].append(item["predicted_class"])
        
    labels = ["Safe", "Harmful"]
    for v, d in variants.items():
        cm = confusion_matrix(d["y_true"], d["y_pred"], labels=labels)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, annot_kws={"size": 16})
        
        acc = sum(1 for t, p in zip(d["y_true"], d["y_pred"]) if t == p) / len(d["y_true"]) * 100
        
        plt.title(f'Variant {v} Confusion Matrix\n(Ground Truth = Refusal Score)\nAccuracy: {acc:.1f}%', pad=20, fontweight="bold")
        plt.ylabel('True Label (via Refusal Score)')
        plt.xlabel('Predicted Label (Internal Judge)')
        
        output_file = os.path.join(plots_dir, f'04_confusion_matrix_{v}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Generated {output_file} | Accuracy: {acc:.1f}%")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    t06_dir = os.path.join(base_dir, "llama_results_620_t06")
    t10_dir = os.path.join(base_dir, "llama_results_620_t10")
    
    print("🚀 Plotting Original Matrices for Temp 0.6...")
    plot_original_confusion_for_dir(t06_dir)
    
    print("\n🚀 Plotting Original Matrices for Temp 1.0...")
    plot_original_confusion_for_dir(t10_dir)
