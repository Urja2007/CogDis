import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax

import config

def run_all_softmax_confidence():
    results_dir = config.get_results_dir()
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    meta_eval_path = os.path.join(results_dir, "meta_eval_confusion.json")
    if not os.path.exists(meta_eval_path):
        print("meta_eval_confusion.json not found.")
        return
        
    with open(meta_eval_path, "r") as f:
        meta_eval = json.load(f)
        
    # Group by prompt ID
    results_by_id = {}
    for item in meta_eval:
        pid = item["id"]
        if pid not in results_by_id:
            results_by_id[pid] = {}
        results_by_id[pid][item["variant"]] = item
        
    variants = ["A1", "A2", "A3", "B1", "B2", "B3"]
    
    for variant in variants:
        confidence_scores = []
        is_correct_list = []
        
        for pid, var_dict in results_by_id.items():
            if variant in var_dict:
                item = var_dict[variant]
                
                # We need to extract the raw logits depending on the variant's key format
                if "logit_refuse" in item and "logit_provide" in item:
                    l1 = item["logit_refuse"]
                    l2 = item["logit_provide"]
                elif "logit_harmful_pos" in item and "logit_safe_neg" in item:
                    l1 = item["logit_harmful_pos"]
                    l2 = item["logit_safe_neg"]
                else:
                    continue 
                    
                logits = np.array([l1, l2])
                probs = softmax(logits)
                
                # The model's confidence in its CHOSEN answer
                confidence = np.max(probs) * 100
                confidence_scores.append(confidence)
                is_correct_list.append(item["is_correct"])
                
        if not confidence_scores:
            print(f"Skipping {variant} - no logit data found.")
            continue
            
        conf_correct = [c for c, is_corr in zip(confidence_scores, is_correct_list) if is_corr]
        conf_incorrect = [c for c, is_corr in zip(confidence_scores, is_correct_list) if not is_corr]
        
        plt.figure(figsize=(10, 6))
        
        if conf_incorrect:
            sns.histplot(conf_incorrect, color="crimson", label="Incorrect Answer", bins=20, binrange=(50, 100), alpha=0.7)
        if conf_correct:
            sns.histplot(conf_correct, color="mediumseagreen", label="Correct Answer", bins=20, binrange=(50, 100), alpha=0.7)
            
        plt.title(f"Ablated Model Confidence Distribution ({variant})", fontsize=16, pad=15)
        plt.xlabel("Softmax Probability (%) of Chosen Answer", fontsize=14)
        plt.ylabel("Number of Prompts", fontsize=14)
        plt.xlim(50, 100)
        plt.legend(fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(plots_dir, f"09_softmax_confidence_histogram_{variant}.png")
        plt.savefig(save_path, dpi=300)
        print(f"Generated {save_path}")

if __name__ == "__main__":
    run_all_softmax_confidence()
