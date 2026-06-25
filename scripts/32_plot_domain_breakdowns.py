import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import config

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_domain_breakdowns():
    print("🚀 GENERATING DOMAIN-WISE BEHAVIOR & RECOGNITION PLOTS 🚀")
    
    results_dir = config.get_results_dir()
    plots_dir = os.path.join(results_dir, "plots", "domain_breakdowns")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load base prompts to map prompt_id to category
    data_dir = os.path.join(config.BASE_DIR, "data")
    with open(os.path.join(data_dir, "harmful_prompts.json"), "r") as f:
        harmful_prompts = json.load(f)
        
    id_to_category = {item.get("prompt_id", item.get("id")): item["category"] for item in harmful_prompts}
    
    # -------------------------------------------------------------
    # 1. Load Generation Data (For Figure 1 & 2)
    # -------------------------------------------------------------
    orig_path = os.path.join(results_dir, "original_generation.json")
    abl_path = os.path.join(results_dir, "ablated_generation.json")
    
    with open(orig_path, "r") as f: orig_data = json.load(f)
    with open(abl_path, "r") as f: abl_data = json.load(f)
        
    df_gen = []
    
    # Helper to process generation files
    def process_gen(data, model_type):
        for item in data:
            if item["prompt_id"] not in id_to_category: continue # Skip harmless for domain breakdown
            cat = id_to_category[item["prompt_id"]]
            
            refusal_score = item.get("refusal_score", 0)
            # LlamaGuard: 1 = Safe, 0 = Unsafe. Unsafe rate = 1 - llamaguard_score
            unsafe_score = 1 - item.get("llamaguard_score", 1)
            
            df_gen.append({
                "Category": cat,
                "Model": model_type,
                "Refusal_Rate": refusal_score * 100,
                "Unsafe_Rate": unsafe_score * 100
            })
            
    process_gen(orig_data, "Original")
    process_gen(abl_data, "Ablated")
    
    df_gen = pd.DataFrame(df_gen)
    
    # Plot Figure 1: Refusal Rate by Domain
    plt.figure(figsize=(14, 7))
    sns.barplot(data=df_gen, x="Category", y="Refusal_Rate", hue="Model", palette=["#2ecc71", "#e74c3c"])
    plt.title("Refusal Rate by Domain (Before vs After Ablation)", pad=20, fontweight="bold")
    plt.ylabel("Refusal Rate (%)")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "fig1_refusal_by_domain.png"), dpi=300)
    plt.close()
    
    # Plot Figure 2: Compliance Severity (Unsafe Rate) by Domain
    plt.figure(figsize=(14, 7))
    sns.barplot(data=df_gen, x="Category", y="Unsafe_Rate", hue="Model", palette=["#2ecc71", "#e74c3c"])
    plt.title("LlamaGuard Unsafe Compliance by Domain (Before vs After Ablation)", pad=20, fontweight="bold")
    plt.ylabel("Unsafe Compliance Rate (%)")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "fig2_compliance_by_domain.png"), dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # 2. Load Harm Recognition Data (For Figure 3)
    # -------------------------------------------------------------
    meta_path = os.path.join(results_dir, "meta_eval_confusion.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta_data = json.load(f)
            
        # We want Variant A1 (The model evaluating the Ablated policy)
        # And Variant B1 (The model evaluating the Original policy? No, we want to compare how the Judge evaluates Harmful text vs Safe text)
        # Actually, the user asked for "Harm Recognition Probability Before vs After Ablation". 
        # But we didn't run the Judge on the Original model's completions because they were refusals! 
        # Wait, the closest thing we have to "Harm Recognition Probability" is comparing the Logit Probabilities of Harmful vs Harmless base prompts!
        # Let's extract Variant A1 for Harmful prompts vs Variant B1?
        # Actually, if we look at Variant A1 on the ablated harmful completions:
        
        df_recog = []
        for item in meta_data:
            if item["variant"] != "A1": continue
            # Check if this prompt was originally from the harmful set
            pid = item.get("prompt_id", item.get("id"))
            if pid not in id_to_category: continue
                
            cat = id_to_category[pid]
            logit_diff = item.get("logit_difference", 0)
            
            # Since true_label = "Harmful", positive logit difference means it thinks it's Harmful.
            prob = sigmoid(logit_diff)
            
            df_recog.append({
                "Category": cat,
                "Harm_Probability": prob
            })
            
        if len(df_recog) > 0:
            df_recog = pd.DataFrame(df_recog)
            mean_probs = df_recog.groupby("Category")["Harm_Probability"].mean().reset_index()
            
            # Since we only have the "After Ablation" (Ablated) model's self-evaluation for A1, we will plot it as a Heatmap
            plt.figure(figsize=(10, 8))
            pivot_probs = mean_probs.set_index("Category")
            sns.heatmap(pivot_probs, annot=True, cmap="YlOrRd", vmin=0, vmax=1, fmt=".2f",
                        cbar_kws={'label': 'Probability of Recognizing Harm'})
            plt.title("Ablated Model's Internal Harm Recognition Probability\n(By Domain on its own Harmful Output)", pad=20, fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "fig3_harm_recognition_heatmap.png"), dpi=300)
            plt.close()
            
    print("✅ Successfully generated all domain breakdown figures!")

if __name__ == "__main__":
    generate_domain_breakdowns()
