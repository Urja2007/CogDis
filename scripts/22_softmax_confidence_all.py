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
                
                confidence = np.max(probs) * 100
                confidence_scores.append(confidence)
                is_correct_list.append(item["is_correct"])
                
        if not confidence_scores:
            continue
            
        conf_correct = [c for c, is_corr in zip(confidence_scores, is_correct_list) if is_corr]
        conf_incorrect = [c for c, is_corr in zip(confidence_scores, is_correct_list) if not is_corr]
        
        plt.figure(figsize=(10, 6))
        
        # Define dynamic labels based on variant type (A = Generation, B = Perception)
        if variant.startswith("A"):
            correct_label = "Correct (Refused the Harmful Request)"
            incorrect_label = "Incorrect (Malicious Compliance)"
            title_type = "Behavioral Output (The Policy)"
            explainer_text = "GREEN = Model successfully refused to help.\nRED = Model happily provided dangerous instructions."
        else:
            correct_label = "Correct (Successfully Recognized Danger)"
            incorrect_label = "Incorrect (Failed to Recognize Danger)"
            title_type = "Latent Perception (The Judge)"
            explainer_text = "GREEN = Model accurately identified the text as dangerous.\nRED = Model was fooled into thinking it was safe."
            
        if conf_incorrect:
            sns.histplot(conf_incorrect, color="crimson", label=incorrect_label, bins=20, binrange=(0, 100), alpha=0.7)
        if conf_correct:
            sns.histplot(conf_correct, color="mediumseagreen", label=correct_label, bins=20, binrange=(0, 100), alpha=0.7)
            
        plt.title(f"Ablated Model Confidence: {title_type}\n(Variant {variant})", fontsize=16, pad=15, fontweight="bold")
        plt.xlabel("Softmax Probability (%) of Chosen Answer", fontsize=14)
        plt.ylabel("Number of Prompts", fontsize=14)
        plt.xlim(0, 100)
        
        # Add the explanation box directly onto the plot
        props = dict(boxstyle='round,pad=0.5', facecolor='whitesmoke', alpha=0.9, edgecolor='gray')
        plt.text(2, plt.ylim()[1]*0.85, explainer_text, fontsize=12, bbox=props, verticalalignment='top')
        
        plt.legend(fontsize=11, loc='upper left', bbox_to_anchor=(0, 0.75))
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(plots_dir, f"09_softmax_confidence_histogram_{variant}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Generated {save_path}")

if __name__ == "__main__":
    run_all_softmax_confidence()
