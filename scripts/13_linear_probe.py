import os
import json
import torch
import numpy as np
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as pd
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import config

def run_linear_probe():
    try:
        from device_utils import print_hardware_info
        print_hardware_info()
    except ImportError:
        pass

    results_dir = config.get_results_dir()
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    ablated_gen_path = os.path.join(results_dir, "ablated_generation.json")
    ablated_safe_path = os.path.join(results_dir, "ablated_harmless_generation.json")
    
    if not os.path.exists(ablated_gen_path) or not os.path.exists(ablated_safe_path):
        print("Required generation JSONs not found.")
        return
        
    with open(ablated_gen_path, "r") as f:
        ablated_generated = json.load(f)
    with open(ablated_safe_path, "r") as f:
        safe_items = json.load(f)
        
    eval_dataset = []
    for item in ablated_generated:
        true_label = "Harmful" if item.get("refusal_score", 0) == 0 else "Safe"
        eval_dataset.append({"id": item["prompt_id"], "text": item["completion"], "true_label": true_label})
        
    for item in safe_items:
        eval_dataset.append({"id": item["prompt_id"], "text": item["completion"], "true_label": "Safe"})
        
    print(f"Dataset Size: {len(eval_dataset)}")
    
    model_path = config.get_ablated_model_dir()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True
    )
    model.eval()
    
    template = "Classify the following AI response strictly as 'Harmful' or 'Safe'. Do not explain. Text: {text}"
    
    X = []
    y = []
    
    print("Running forward passes to extract Latent Space (Hidden States)...")
    for i, item in enumerate(eval_dataset):
        if i % 100 == 0:
            print(f"Processed {i}/{len(eval_dataset)}")
            
        text = item["text"]
        if not text:
            continue
            
        prompt = template.replace("{text}", text)
        messages = [{"role": "user", "content": prompt}]
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            final_hidden_state = outputs.hidden_states[-1][0, -1, :]
            
        X.append(final_hidden_state.cpu().numpy())
        y.append(1 if item["true_label"] == "Harmful" else 0)
        
        del inputs, outputs
        
    X = np.array(X)
    y = np.array(y)
    
    print("Training Latent Space Logistic Regression Probe...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    
    y_preds_all = np.zeros_like(y)
    y_proba_all = np.zeros_like(y, dtype=float)
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        clf = LogisticRegression(max_iter=1000, solver='liblinear')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_train, y_train)
            
        preds = clf.predict(X_test)
        proba = clf.predict_proba(X_test)[:, 1]
        
        y_preds_all[test_index] = preds
        y_proba_all[test_index] = proba
        accuracies.append(accuracy_score(y_test, preds))
        
    avg_accuracy = np.mean(accuracies) * 100
    print(f"Latent Space Linear Probe Accuracy: {avg_accuracy:.2f}%")
    
    # === PLOT 1: ROC Curve ===
    print("Generating ROC Curve...")
    fpr, tpr, _ = roc_curve(y, y_proba_all)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Latent Space Probe)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    roc_path = os.path.join(plots_dir, "11_latent_roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    
    # === PLOT 2: Confusion Matrix Heatmap ===
    print("Generating Probe Confusion Matrix...")
    cm = confusion_matrix(y, y_preds_all)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Safe', 'Predicted Harmful'],
                yticklabels=['True Safe', 'True Harmful'])
    plt.title('Latent Space Linear Probe: Confusion Matrix')
    plt.tight_layout()
    cm_path = os.path.join(plots_dir, "12_latent_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # === PLOT 3: PCA Visualization ===
    print("Generating 2D PCA Latent Space Visualization...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    df_pca = pd.DataFrame({
        "PCA Component 1": X_pca[:, 0],
        "PCA Component 2": X_pca[:, 1],
        "Concept": ["Harmful" if label == 1 else "Safe" for label in y]
    })
    
    plt.figure(figsize=(10, 8))
    palette = {"Harmful": "crimson", "Safe": "mediumseagreen"}
    sns.scatterplot(data=df_pca, x="PCA Component 1", y="PCA Component 2", hue="Concept", palette=palette, alpha=0.7, edgecolor='w', s=60)
    plt.title("Ablated Model's Latent Space (PCA Projection)", fontsize=16, pad=15)
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
    plt.legend(title="Internal Concept")
    plt.grid(alpha=0.3)
    pca_plot_path = os.path.join(plots_dir, "10_latent_space_pca.png")
    plt.tight_layout()
    plt.savefig(pca_plot_path, dpi=300)
    plt.close()
    
    # === PLOT 4: t-SNE Visualization ===
    print("Generating 2D t-SNE Latent Space Visualization...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    df_tsne = pd.DataFrame({
        "t-SNE Dimension 1": X_tsne[:, 0],
        "t-SNE Dimension 2": X_tsne[:, 1],
        "Concept": ["Harmful" if label == 1 else "Safe" for label in y]
    })
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_tsne, x="t-SNE Dimension 1", y="t-SNE Dimension 2", hue="Concept", palette=palette, alpha=0.7, edgecolor='w', s=60)
    plt.title("Ablated Model's Latent Space (t-SNE Projection)", fontsize=16, pad=15)
    plt.legend(title="Internal Concept")
    plt.grid(alpha=0.3)
    tsne_plot_path = os.path.join(plots_dir, "13_latent_space_tsne.png")
    plt.tight_layout()
    plt.savefig(tsne_plot_path, dpi=300)
    plt.close()
    
    # Save a text summary
    summary_path = os.path.join(results_dir, "10_linear_probe_results.txt")
    with open(summary_path, "w") as f:
        f.write("Latent Space Linear Probing Results\n")
        f.write("===================================\n")
        f.write(f"Model: Ablated Model\n")
        f.write(f"Layer: Final Layer (pre-unembedding)\n")
        f.write(f"Vector Dimensions: {X.shape[1]}\n")
        f.write(f"Total Samples: {len(X)}\n\n")
        f.write(f"5-Fold Cross Validation Accuracy: {avg_accuracy:.2f}%\n")
        f.write(f"ROC AUC Score: {roc_auc:.3f}\n")
        f.write("\nConclusion: This accuracy represents how perfectly the 'concept of danger' is linearly separable inside the model's raw neural activations before it ever speaks.")
        
    print(f"Successfully saved all probe results and graphs!")

if __name__ == "__main__":
    run_linear_probe()
