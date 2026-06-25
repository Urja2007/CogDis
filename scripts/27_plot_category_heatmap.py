import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import config

def plot_category_heatmap():
    results_dir = config.get_results_dir()
    plots_dir = os.path.join(results_dir, "plots", "category_heatmap")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Load Categories
    harmful_path = os.path.join(config.BASE_DIR, "data", "harmful_prompts.json")
    with open(harmful_path, "r") as f:
        harmful_data = json.load(f)
        
    category_map = {item["id"]: item.get("category", "Unknown") for item in harmful_data}
    
    # 2. Load Generations
    ablated_path = os.path.join(results_dir, "ablated_generation.json")
    if not os.path.exists(ablated_path):
        print(f"File not found: {ablated_path}")
        return
        
    with open(ablated_path, "r") as f:
        gen_data = json.load(f)
        
    # Check if llamaguard score exists
    if not gen_data or "llamaguard_score" not in gen_data[0]:
        print("Missing LlamaGuard scores.")
        return
        
    # 3. Aggregate Data
    results = []
    for item in gen_data:
        pid = item["prompt_id"]
        cat = category_map.get(pid, "Unknown")
        
        ref = item.get("refusal_score", 0)  # 0 = Complied, 1 = Refused
        lg = item.get("llamaguard_score", 0) # 0 = Unsafe, 1 = Safe
        
        if ref == 1 and lg == 1:
            quadrant = "Refused & Safe"
        elif ref == 0 and lg == 1:
            quadrant = "Complied & Safe\n(Edge Case)"
        elif ref == 1 and lg == 0:
            quadrant = "Refused & Unsafe\n(Edge Case)"
        else:
            quadrant = "Complied & Unsafe"
            
        results.append({
            "Category": cat,
            "Quadrant": quadrant
        })
        
    df = pd.DataFrame(results)
    
    # Calculate percentages
    grouped = df.groupby(["Category", "Quadrant"]).size().unstack(fill_value=0)
    
    # Normalize by row (category) to get percentages
    percentages = grouped.div(grouped.sum(axis=1), axis=0) * 100
    
    # Ensure all 4 columns exist for consistent plotting
    for q in ["Complied & Unsafe", "Complied & Safe\n(Edge Case)", "Refused & Unsafe\n(Edge Case)", "Refused & Safe"]:
        if q not in percentages.columns:
            percentages[q] = 0.0
            
    # Reorder columns
    cols_order = ["Complied & Unsafe", "Complied & Safe\n(Edge Case)", "Refused & Unsafe\n(Edge Case)", "Refused & Safe"]
    percentages = percentages[cols_order]
    
    # Create custom annotations with both Percentage and Count
    annot_data = pd.DataFrame(index=percentages.index, columns=percentages.columns)
    for col in percentages.columns:
        for idx in percentages.index:
            pct = percentages.at[idx, col]
            count = grouped.at[idx, col] if col in grouped.columns else 0
            annot_data.at[idx, col] = f"{pct:.1f}%\n(n={count})"
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        percentages, 
        annot=annot_data, 
        fmt="", 
        cmap="YlOrRd", 
        linewidths=0.5,
        cbar_kws={'label': 'Percentage of Prompts (%)'}
    )
    
    plt.title(f"Ablated Model: Outcome Distribution by Domain\n({os.path.basename(results_dir)})", pad=20, fontsize=14, fontweight="bold")
    plt.ylabel("Prompt Category (Domain)", labelpad=15, fontsize=12)
    plt.xlabel("Behavioral Outcome Quadrant", labelpad=15, fontsize=12)
    
    plt.tight_layout()
    save_path = os.path.join(plots_dir, "domain_heatmap.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Generated {save_path}")

if __name__ == "__main__":
    plot_category_heatmap()
