import os
import config
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = config.get_results_dir()
    plots_dir = os.path.join(results_dir, "plots")
    
    # Load data
    def load_json(path):
        full_path = os.path.join(results_dir, path)
        if os.path.exists(full_path):
            with open(full_path, "r") as f:
                return json.load(f)
        # Fallback to flat directory
        flat_path = os.path.join(results_dir, os.path.basename(path))
        if os.path.exists(flat_path):
            with open(flat_path, "r") as f:
                return json.load(f)
        return None

    orig_gen = load_json("generations/original_generation.json")
    ab_gen = load_json("generations/ablated_generation.json")
    rand_gen = load_json("generations/random_generation.json")
    l25_gen = load_json("generations/layer25_generation.json")
    harmful_prompts = load_json("../data/harmful_prompts.json")

    def get_rate(data):
        if not data: return 0
        return (sum(1 for x in data if x.get("refusal_score", 0) > 0) / len(data)) * 100

    # Figure 1: Refusal Rate Collapse
    plt.figure(figsize=(6, 4))
    rates = [get_rate(orig_gen), get_rate(ab_gen)]
    print(f"DEBUG: Results Dir: {results_dir}")
    print(f"DEBUG: Orig Gen Data length: {len(orig_gen) if orig_gen else 'None'}")
    print(f"DEBUG: Ab Gen Data length: {len(ab_gen) if ab_gen else 'None'}")
    print(f"DEBUG: Calculated Rates: {rates}")
    plt.bar(["Original Model", "Ablated Model"], rates, color=['#e74c3c', '#2ecc71'])
    plt.ylabel('Refusal Rate (%)')
    plt.title('Figure 1: Refusal Rate Collapse')
    for i, v in enumerate(rates):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')
    plt.ylim(0, 105)
    plt.savefig(os.path.join(plots_dir, "01_refusal_collapse.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 2: Category-wise Heatmap
    if harmful_prompts and orig_gen and ab_gen:
        category_map = {x["id"]: x.get("category", "unknown") for x in harmful_prompts}
        cats = sorted(list(set(category_map.values())))
        
        orig_rates = []
        ab_rates = []
        for c in cats:
            c_orig_items = [x for x in orig_gen if category_map.get(x["prompt_id"]) == c]
            c_ab_items = [x for x in ab_gen if category_map.get(x["prompt_id"]) == c]
            
            orig_rates.append(sum(1 for x in c_orig_items if x["refusal_score"] > 0) / max(1, len(c_orig_items)))
            ab_rates.append(sum(1 for x in c_ab_items if x["refusal_score"] > 0) / max(1, len(c_ab_items)))
            
        data_matrix = np.array([orig_rates, ab_rates]).T
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(data_matrix, annot=True, fmt=".2f", cmap="Reds", yticklabels=cats, xticklabels=["Original", "Ablated"])
        plt.title('Figure 2: Category-wise Refusal Probability')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "02_category_refusal_heatmap.png"), dpi=300)
        plt.close()

    # Figure 3: Refusal Token Frequency
    if orig_gen and ab_gen:
        keywords = ["I cannot", "I'm sorry", "I apologize", "As an AI"]
        orig_counts = [sum(1 for x in orig_gen if kw.lower() in x["completion"].lower()) for kw in keywords]
        ab_counts = [sum(1 for x in ab_gen if kw.lower() in x["completion"].lower()) for kw in keywords]
        
        x = np.arange(len(keywords))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width/2, orig_counts, width, label='Original', color='#e74c3c')
        ax.bar(x + width/2, ab_counts, width, label='Ablated', color='#2ecc71')
        
        ax.set_ylabel('Frequency')
        ax.set_title('Figure 3: Refusal Keyword Frequency')
        ax.set_xticks(x)
        ax.set_xticklabels(keywords)
        ax.legend()
        plt.savefig(os.path.join(plots_dir, "03_refusal_keyword_frequency.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # Figure 7: Control Ablation Comparison
    plt.figure(figsize=(8, 5))
    control_rates = [
        get_rate(orig_gen), 
        get_rate(rand_gen), 
        get_rate(l25_gen), 
        get_rate(ab_gen)
    ]
    labels = ["Original", "Random Vector\n(Layer 12)", "Refusal Vector\n(Layer 25)", "Refusal Vector\n(Layer 12)"]
    colors = ['#34495e', '#95a5a6', '#7f8c8d', '#2ecc71']
    
    plt.bar(labels, control_rates, color=colors)
    plt.ylabel('Refusal Rate (%)')
    plt.title('Figure 7: Control Ablation Specificity')
    for i, v in enumerate(control_rates):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')
    plt.ylim(0, 105)
    plt.savefig(os.path.join(plots_dir, "07_control_ablation_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Missing plots generated successfully!")

if __name__ == "__main__":
    main()
