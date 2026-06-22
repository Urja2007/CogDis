import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
import config
from sklearn.metrics import confusion_matrix

def generate_plots():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = config.get_results_dir()
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # --- Check files exist ---
    stage1_orig = os.path.join(results_dir, "original_generation.json")
    stage1_ablated = os.path.join(results_dir, "ablated_generation.json")
    stage2_orig = os.path.join(results_dir, "meta_eval_original.json")
    stage2_ablated = os.path.join(results_dir, "meta_eval_ablated.json")
    
    if not all(os.path.exists(p) for p in [stage1_orig, stage1_ablated, stage2_orig, stage2_ablated]):
        print("Missing Stage 1 or Stage 2 JSON results. Run Phase 6 & 7 first!")
        return

    # --- Plot 1: Cognitive Dissociation Bar Chart ---
    with open(stage1_orig, "r") as f:
        orig_gen = json.load(f)
    with open(stage1_ablated, "r") as f:
        ablated_gen = json.load(f)
        
    orig_refusal_rate = sum(1 for x in orig_gen if x["refusal_score"] > 0) / len(orig_gen)
    ablated_refusal_rate = sum(1 for x in ablated_gen if x["refusal_score"] > 0) / len(ablated_gen)
    
    with open(stage2_orig, "r") as f:
        meta_orig = json.load(f)
    with open(stage2_ablated, "r") as f:
        meta_ablated = json.load(f)
        
    meta_orig_A = [x for x in meta_orig if x["variant"] == "A"]
    meta_ablated_A = [x for x in meta_ablated if x["variant"] == "A"]
    
    orig_acc = sum(1 for x in meta_orig_A if x["is_correct"]) / max(1, len(meta_orig_A))
    ablated_acc = sum(1 for x in meta_ablated_A if x["is_correct"]) / max(1, len(meta_ablated_A))
    
    plt.figure(figsize=(8, 6))
    bar_width = 0.35
    x = [0, 1]
    
    refusal_rates = [orig_refusal_rate * 100, ablated_refusal_rate * 100]
    acc_rates = [orig_acc * 100, ablated_acc * 100]
    
    plt.bar([p - bar_width/2 for p in x], refusal_rates, width=bar_width, color='#3498db', label='Refusal Rate (Behavior)')
    plt.bar([p + bar_width/2 for p in x], acc_rates, width=bar_width, color='#e74c3c', label='Harm Recognition (Understanding)')
    
    plt.xticks(x, ['Original Model', 'Ablated Model'])
    plt.ylabel('Percentage (%)')
    plt.title('Cognitive Dissociation: Behavior vs Understanding')
    plt.legend(loc="upper right")
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(plots_dir, "01_refusal_vs_recognition.png"), dpi=300)
    plt.close()
    
    # --- Plot 2: Logit Distribution ---
    plt.figure(figsize=(8, 6))
    orig_logits = [x["logit_difference"] for x in meta_orig_A]
    ablated_logits = [x["logit_difference"] for x in meta_ablated_A]
    
    sns.kdeplot(orig_logits, fill=True, color='#3498db', label='Original Model')
    sns.kdeplot(ablated_logits, fill=True, color='#e74c3c', label='Ablated Model')
    plt.axvline(0, color='black', linestyle='--')
    plt.xlabel('Logit Difference (Harmful - Safe)')
    plt.ylabel('Density')
    plt.title('Confidence in Classifying Harm (Meta-Evaluation)')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "02_logit_distribution.png"), dpi=300)
    plt.close()
    
    # --- Plot 3: Layerwise Re-emergence ---
    layerwise_path = os.path.join(results_dir, "layerwise_cosine.json")
    if os.path.exists(layerwise_path):
        with open(layerwise_path, "r") as f:
            layerwise = json.load(f)
            
        gen_layers = [x["layer"] for x in layerwise["generation_layerwise"]]
        gen_cos = [x["mean_cos"] for x in layerwise["generation_layerwise"]]
        gen_std = [x["std"] for x in layerwise["generation_layerwise"]]
        
        eval_layers = [x["layer"] for x in layerwise["meta_evaluation_layerwise"]]
        eval_cos = [x["mean_cos"] for x in layerwise["meta_evaluation_layerwise"]]
        eval_std = [x["std"] for x in layerwise["meta_evaluation_layerwise"]]
        
        plt.figure(figsize=(10, 6))
        plt.plot(gen_layers, gen_cos, label='Stage 1: Generating Harm', color='#3498db', linewidth=2)
        plt.fill_between(gen_layers, [c - s for c, s in zip(gen_cos, gen_std)], [c + s for c, s in zip(gen_cos, gen_std)], color='#3498db', alpha=0.2)
        
        plt.plot(eval_layers, eval_cos, label='Stage 2: Evaluating Harm', color='#e74c3c', linewidth=2)
        plt.fill_between(eval_layers, [c - s for c, s in zip(eval_cos, eval_std)], [c + s for c, s in zip(eval_cos, eval_std)], color='#e74c3c', alpha=0.2)
        
        plt.axhline(0, color='black', linestyle=':', alpha=0.5)
        plt.xlabel('Layer Number')
        plt.ylabel('Cosine Similarity to Refusal Vector')
        plt.title('Layer-wise Re-emergence of Harm Concept')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, "03_layerwise_reemergence.png"), dpi=300)
        plt.close()
        print("Generated 03_layerwise_reemergence.png")

    # --- Plot 4: Confusion Matrix ---
    confusion_path = os.path.join(results_dir, "meta_eval_confusion.json")
    if os.path.exists(confusion_path):
        with open(confusion_path, "r") as f:
            confusion_data = json.load(f)
            
        # Filter for Variant A (Direct classification)
        data_A = [x for x in confusion_data if x["variant"] == "A"]
        
        y_true = [x["true_label"] for x in data_A]
        y_pred = [x["predicted_class"] for x in data_A]
        
        labels = ["Harmful", "Safe"]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=False, annot_kws={"size": 16})
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('Actual Label', fontsize=12)
        plt.title('Ablated Model: Semantic Harm Recognition\nConfusion Matrix', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "04_confusion_matrix.png"), dpi=300)
        plt.close()
        print("Generated 04_confusion_matrix.png")

    print("Generated 01_refusal_vs_recognition.png")
    print("Generated 02_logit_distribution.png")
    print("All plots generated successfully in results/plots/")

if __name__ == "__main__":
    try:
        from device_utils import print_hardware_info
        print_hardware_info()
    except ImportError:
        pass
    generate_plots()
