import json
import os
import config
import numpy as np

def generate_tables():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    results_dir = config.get_results_dir()
    out_file = os.path.join(results_dir, "paper_tables_and_stats.md")
    
    output = ["# Cognitive Dissociation: Analysis Tables & Statistics\n"]
    
    # Helper to load JSON safely
    def load_json(path):
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return None

    # Load all available data
    harmful_prompts = load_json(os.path.join(data_dir, "harmful_prompts.json"))
    orig_gen = load_json(os.path.join(results_dir, "original_generation.json"))
    ablated_gen = load_json(os.path.join(results_dir, "ablated_generation.json"))
    meta_orig = load_json(os.path.join(results_dir, "meta_eval_original.json"))
    meta_ablated = load_json(os.path.join(results_dir, "meta_eval_ablated.json"))
    meta_conf = load_json(os.path.join(results_dir, "meta_eval_confusion.json"))
    layerwise = load_json(os.path.join(results_dir, "layerwise_cosine.json"))
    
    # --- 1.1 Dataset Statistics ---
    output.append("## 1.1 Dataset Statistics")
    if harmful_prompts:
        categories = set([x.get("category", "unknown") for x in harmful_prompts])
        output.append(f"- **Harmful Instructions**: {len(harmful_prompts)} items")
        output.append(f"- **Categories Present**: {', '.join(categories)}\n")

    # --- 2.1 Refusal Rate Matrix ---
    output.append("## 2.1 Refusal Rate Matrix")
    output.append("| Model | Refused | Complied | Refusal Rate |")
    output.append("|-------|---------|----------|--------------|")
    
    if orig_gen:
        refused = sum(1 for x in orig_gen if x["refusal_score"] > 0)
        complied = len(orig_gen) - refused
        rate = (refused / len(orig_gen)) * 100
        output.append(f"| Original Llama-3-8B | {refused} | {complied} | {rate:.1f}% |")
        
    if ablated_gen:
        refused = sum(1 for x in ablated_gen if x["refusal_score"] > 0)
        complied = len(ablated_gen) - refused
        rate = (refused / len(ablated_gen)) * 100
        output.append(f"| Ablated Model       | {refused} | {complied} | {rate:.1f}% |")
    output.append("\n")

    # --- 2.3 Refusal Category Breakdown ---
    output.append("## 2.3 Refusal Category Breakdown")
    output.append("| Category | Original Refusal Rate | Ablated Refusal Rate |")
    output.append("|----------|-----------------------|----------------------|")
    
    if harmful_prompts and orig_gen and ablated_gen:
        category_map = {x["id"]: x.get("category", "unknown") for x in harmful_prompts}
        cats = sorted(list(set(category_map.values())))
        
        for c in cats:
            # Original
            c_orig_items = [x for x in orig_gen if category_map.get(x["prompt_id"]) == c]
            c_orig_rate = (sum(1 for x in c_orig_items if x["refusal_score"] > 0) / max(1, len(c_orig_items))) * 100
            
            # Ablated
            c_ab_items = [x for x in ablated_gen if category_map.get(x["prompt_id"]) == c]
            c_ab_rate = (sum(1 for x in c_ab_items if x["refusal_score"] > 0) / max(1, len(c_ab_items))) * 100
            
            output.append(f"| {c} | {c_orig_rate:.1f}% | {c_ab_rate:.1f}% |")
    output.append("\n")

    # --- 2.4 Refusal Keyword Distribution ---
    output.append("## 2.4 Refusal Keyword Distribution")
    output.append("Frequency of refusal keywords in generated text:\n")
    keywords = ["I cannot", "I'm sorry", "I apologize", "As an AI"]
    output.append("| Keyword | Original Model | Ablated Model |")
    output.append("|---------|----------------|---------------|")
    if orig_gen and ablated_gen:
        for kw in keywords:
            kw_lower = kw.lower()
            orig_count = sum(1 for x in orig_gen if kw_lower in x["completion"].lower())
            ab_count = sum(1 for x in ablated_gen if kw_lower in x["completion"].lower())
            output.append(f"| '{kw}' | {orig_count} | {ab_count} |")
    output.append("\n")

    # --- 3.1 & 3.4 Classification Accuracy / Self Recognition ---
    output.append("## 3.1 & 3.4 Harm Recognition Accuracy (Variant A - Direct)")
    output.append("| Model | Accuracy |")
    output.append("|-------|----------|")
    if meta_orig:
        meta_orig_A = [x for x in meta_orig if x["variant"] == "A"]
        orig_acc = (sum(1 for x in meta_orig_A if x["is_correct"]) / max(1, len(meta_orig_A))) * 100
        output.append(f"| Original Llama-3-8B | {orig_acc:.1f}% |")
        
    if meta_ablated:
        meta_ablated_A = [x for x in meta_ablated if x["variant"] == "A"]
        ablated_acc = (sum(1 for x in meta_ablated_A if x["is_correct"]) / max(1, len(meta_ablated_A))) * 100
        output.append(f"| Ablated Model       | {ablated_acc:.1f}% |")
    output.append("\n")

    # --- 3.2 & 3.3 Confusion Matrix & F1 ---
    output.append("## 3.2 & 3.3 Confusion Matrix & Precision/Recall/F1 (Ablated Model)")
    if meta_conf:
        meta_conf_A = [x for x in meta_conf if x["variant"] == "A"]
        
        tp = sum(1 for x in meta_conf_A if x["true_label"] == "Harmful" and x["predicted_class"] == "Harmful")
        fn = sum(1 for x in meta_conf_A if x["true_label"] == "Harmful" and x["predicted_class"] == "Safe")
        fp = sum(1 for x in meta_conf_A if x["true_label"] == "Safe" and x["predicted_class"] == "Harmful")
        tn = sum(1 for x in meta_conf_A if x["true_label"] == "Safe" and x["predicted_class"] == "Safe")
        
        output.append("### Confusion Matrix")
        output.append("| Actual \ Predicted | Predicted Harmful | Predicted Safe |")
        output.append("|--------------------|-------------------|----------------|")
        output.append(f"| **Actual Harmful** | {tp} (TP)         | {fn} (FN)      |")
        output.append(f"| **Actual Safe**    | {fp} (FP)         | {tn} (TN)      |")
        output.append("\n")
        
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * (precision * recall) / max(1e-9, precision + recall)
        
        output.append("### Metrics")
        output.append(f"- **Precision**: {precision:.3f}")
        output.append(f"- **Recall**: {recall:.3f}")
        output.append(f"- **F1 Score**: {f1:.3f}")
    else:
        output.append("*Run `06b_confusion_matrix_eval.py` first to generate this data.*")
    output.append("\n")

    # --- 5.1 Refusal Direction Projection (Cosine Similarity) ---
    output.append("## 5.1 Refusal Direction Projection (Cosine Similarity at Key Layers)")
    output.append("| Layer | Generating Harm (Stage 1) | Evaluating Harm (Stage 2) |")
    output.append("|-------|---------------------------|---------------------------|")
    if layerwise:
        gen_data = {x["layer"]: x["mean_cos"] for x in layerwise.get("generation_layerwise", [])}
        eval_data = {x["layer"]: x["mean_cos"] for x in layerwise.get("meta_evaluation_layerwise", [])}
        
        for l in [0, 12, 20, 25, 31]:
            if l in gen_data and l in eval_data:
                output.append(f"| {l} | {gen_data[l]:.4f} | {eval_data[l]:.4f} |")
    output.append("\n")

    # --- 6.1 Harmful vs Safe Logit Difference ---
    output.append("## 6.1 Harmful vs Safe Logit Difference (Variant A)")
    output.append("| Model | Mean Δlogit |")
    output.append("|-------|-------------|")
    if meta_orig:
        meta_orig_A = [x for x in meta_orig if x["variant"] == "A"]
        mean_orig = np.mean([x["logit_difference"] for x in meta_orig_A])
        output.append(f"| Original Llama-3-8B | {mean_orig:.2f} |")
        
    if meta_ablated:
        meta_ablated_A = [x for x in meta_ablated if x["variant"] == "A"]
        mean_ab = np.mean([x["logit_difference"] for x in meta_ablated_A])
        output.append(f"| Ablated Model       | {mean_ab:.2f} |")
    output.append("\n")

    # --- 7.1 & 7.2 Controls: Random Vector & Layer 25 ---
    output.append("## 7.1 & 7.2 Control Experiments (Refusal Rates)")
    output.append("| Model Variant | Refusal Rate |")
    output.append("|---------------|--------------|")
    
    random_gen = load_json(os.path.join(results_dir, "random_generation.json"))
    layer25_gen = load_json(os.path.join(results_dir, "layer25_generation.json"))
    
    if orig_gen:
        rate = sum(1 for x in orig_gen if x["refusal_score"] > 0) / len(orig_gen) * 100
        output.append(f"| Original Llama-3-8B | {rate:.1f}% |")
    if ablated_gen:
        rate = sum(1 for x in ablated_gen if x["refusal_score"] > 0) / len(ablated_gen) * 100
        output.append(f"| Ablated (Layer 12)  | {rate:.1f}% |")
    if random_gen:
        rate = sum(1 for x in random_gen if x["refusal_score"] > 0) / len(random_gen) * 100
        output.append(f"| Random Vector Ablation | {rate:.1f}% |")
    if layer25_gen:
        rate = sum(1 for x in layer25_gen if x["refusal_score"] > 0) / len(layer25_gen) * 100
        output.append(f"| Ablated (Layer 25)  | {rate:.1f}% |")
    output.append("\n")

    # --- 7.3 Paraphrased Prompts ---
    output.append("## 7.3 Control: Paraphrased JailbreakBench Prompts")
    output.append("| Model | Refusal Rate |")
    output.append("|-------|--------------|")
    para_orig = load_json(os.path.join(results_dir, "paraphrased_original_generation.json"))
    para_ablated = load_json(os.path.join(results_dir, "paraphrased_ablated_generation.json"))
    
    if para_orig:
        rate = sum(1 for x in para_orig if x["refusal_score"] > 0) / len(para_orig) * 100
        output.append(f"| Original Llama-3-8B | {rate:.1f}% |")
    if para_ablated:
        rate = sum(1 for x in para_ablated if x["refusal_score"] > 0) / len(para_ablated) * 100
        output.append(f"| Ablated Model       | {rate:.1f}% |")
    output.append("\n")

    # --- 7.4 External Harm ---
    output.append("## 7.4 Control: External Harm (PKU-SafeRLHF) Classification")
    output.append("| Model | Accuracy (Variant A) |")
    output.append("|-------|----------|")
    pku_meta_orig = load_json(os.path.join(results_dir, "meta_eval_pku_original.json"))
    pku_meta_ablated = load_json(os.path.join(results_dir, "meta_eval_pku_ablated.json"))
    
    if pku_meta_orig:
        acc = sum(1 for x in pku_meta_orig if x["is_correct"]) / max(1, len(pku_meta_orig)) * 100
        output.append(f"| Original Llama-3-8B | {acc:.1f}% |")
    if pku_meta_ablated:
        acc = sum(1 for x in pku_meta_ablated if x["is_correct"]) / max(1, len(pku_meta_ablated)) * 100
        output.append(f"| Ablated Model       | {acc:.1f}% |")
    output.append("\n")

    # --- 9.2 Weight Modification Stats ---
    output.append("## 9.2 Weight Modification Statistics (Layer 12)")
    output.append("| Matrix | L2 Norm Difference (ΔW) |")
    output.append("|--------|-------------------------|")
    weight_stats = load_json(os.path.join(results_dir, "weight_modification_stats.json"))
    if weight_stats:
        o_proj = weight_stats[0].get("l2_change", 0)
        down_proj = weight_stats[1].get("l2_change", 0)
        output.append(f"| o_proj | {o_proj:.4f} |")
        output.append(f"| down_proj | {down_proj:.4f} |")
    else:
        output.append("*Run 13_ablation_verification.py first.*")
    output.append("\n")

    # Write to file
    with open(out_file, "w", encoding='utf-8') as f:
        f.write("\n".join(output))
        
    print(f"Successfully generated all available tables and stats!")
    print(f"Saved to: {out_file}")

if __name__ == "__main__":
    generate_tables()
