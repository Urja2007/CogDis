import os
import json
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.special import expit # Sigmoid function

import config

def generate_softmax_histogram():
    results_dir = os.path.join(config.BASE_DIR, config.RESULTS_DIR_NAME)
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    input_file = os.path.join(results_dir, "meta_eval_confusion.json")
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return
        
    with open(input_file, "r") as f:
        data = json.load(f)
        
    # We will focus on Variant B1 for the cleanest understanding plot
    b1_data = [d for d in data if d["variant"] == "B1"]
    
    # Calculate Softmax Probability for each evaluation
    # expit(x) is exactly 1 / (1 + exp(-x))
    probabilities = []
    correct_statuses = []
    
    for d in b1_data:
        # Convert logit_difference to a percentage (0 to 100)
        prob = expit(d["logit_difference"]) * 100
        probabilities.append(prob)
        correct_statuses.append(d["is_correct"])
        
    df = pd.DataFrame({
        "Probability of Harmful (%)": probabilities,
        "Prediction Status": ["Correct" if c else "Incorrect (Error)" for c in correct_statuses]
    })
    
    plt.figure(figsize=(10, 6))
    
    # Custom color palette matching our previous graphs
    palette = {"Correct": "mediumseagreen", "Incorrect (Error)": "crimson"}
    
    sns.histplot(
        data=df,
        x="Probability of Harmful (%)",
        hue="Prediction Status",
        palette=palette,
        multiple="stack",
        bins=20,          # 5% buckets
        binrange=(0, 100),
        edgecolor='white',
        linewidth=1.2
    )
    
    plt.axvline(50, color='black', linestyle='--', alpha=0.8, label='Decision Boundary (50%)')
    
    plt.title('Ablated Model Confidence Histogram (Variant B1 Softmax)', fontsize=14, pad=15)
    plt.xlabel("Model Confidence (0% = Safe, 100% = Harmful)", fontsize=12)
    plt.ylabel("Number of Prompts", fontsize=12)
    
    # Annotate the "Unsure" zone (40% to 60%)
    plt.axvspan(40, 60, color='gray', alpha=0.15, label='Unsure Zone (40-60%)')
    
    plt.legend(title="", loc='upper center')
    
    plt.tight_layout()
    hist_path = os.path.join(plots_dir, "09_softmax_confidence_histogram.png")
    plt.savefig(hist_path, dpi=300)
    print(f"Successfully generated {hist_path}")

if __name__ == "__main__":
    generate_softmax_histogram()
