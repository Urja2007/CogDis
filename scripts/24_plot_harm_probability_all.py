import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.special import expit

import config

def plot_harm_probability():
    results_dir = config.get_results_dir()
    plots_dir = os.path.join(results_dir, "plots", "harm_probability")
    os.makedirs(plots_dir, exist_ok=True)
    
    input_file = os.path.join(results_dir, "meta_eval_confusion.json")
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return
        
    with open(input_file, "r") as f:
        data = json.load(f)
        
    variants = ["A1", "A2", "A3", "B1", "B2", "B3"]
    
    for variant in variants:
        v_data = [d for d in data if d["variant"] == variant]
        if not v_data:
            continue
            
        probabilities = []
        correct_statuses = []
        
        for d in v_data:
            if "logit_difference" in d:
                # Convert logit_difference to a percentage (0 to 100)
                prob = expit(d["logit_difference"]) * 100
                probabilities.append(prob)
                correct_statuses.append(d["is_correct"])
            
        if not probabilities:
            continue
            
        df = pd.DataFrame({
            "Probability of Harmful (%)": probabilities,
            "Prediction Status": ["Correct" if c else "Incorrect (Error)" for c in correct_statuses]
        })
        
        plt.figure(figsize=(10, 6))
        
        # Custom color palette
        palette = {"Correct": "mediumseagreen", "Incorrect (Error)": "crimson"}
        
        # Define dynamic labels based on variant type
        if variant.startswith("A"):
            title_type = "Behavioral Output (The Policy)"
            explainer_text = "GREEN = Model successfully refused to help.\nRED = Model happily provided dangerous instructions."
        else:
            title_type = "Latent Perception (The Judge)"
            explainer_text = "GREEN = Model accurately identified the text as dangerous.\nRED = Model was fooled into thinking it was safe."
            
        sns.histplot(
            data=df,
            x="Probability of Harmful (%)",
            hue="Prediction Status",
            palette=palette,
            multiple="stack",
            bins=20,
            binrange=(0, 100),
            edgecolor='white',
            linewidth=1.2
        )
        
        plt.axvline(50, color='black', linestyle='--', alpha=0.8, label='Decision Boundary (50%)')
        plt.title(f"Probability of Harm Concept: {title_type}\n(Variant {variant})", fontsize=16, pad=15, fontweight="bold")
        plt.xlabel("Model's Perceived Probability of 'Harmful' (%)", fontsize=14)
        plt.ylabel("Number of Prompts", fontsize=14)
        plt.xlim(0, 100)
        
        # Add the explanation box directly onto the plot
        props = dict(boxstyle='round,pad=0.5', facecolor='whitesmoke', alpha=0.9, edgecolor='gray')
        plt.text(2, plt.ylim()[1]*0.85, explainer_text, fontsize=12, bbox=props, verticalalignment='top')
        
        # Adjust legend position so it doesn't overlap with the text box
        plt.legend(fontsize=11, loc='upper left', bbox_to_anchor=(0, 0.75))
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(plots_dir, f"old_style_harm_probability_{variant}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Generated {save_path}")

if __name__ == "__main__":
    plot_harm_probability()
