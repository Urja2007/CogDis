import torch
import os
import matplotlib.pyplot as plt
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
unsupervised_path = os.path.join(base_dir, "unsupervised_refusal_direction.pt")
supervised_path = os.path.join(base_dir, "llama_results_620", "refusal_direction.pt")

u_vec = torch.load(unsupervised_path).float().cpu().squeeze()
s_vec = torch.load(supervised_path).float().cpu().squeeze()

cos_sim = torch.nn.functional.cosine_similarity(u_vec.unsqueeze(0), s_vec.unsqueeze(0)).item()

# Plot
plt.figure(figsize=(8, 6))
bars = plt.bar(['Supervised vs Supervised\n(Baseline)', 'Unsupervised vs Supervised\n(Contrastive Pairs)'], [1.0, cos_sim], color=['#1f77b4', '#d62728'])

plt.ylabel('Cosine Similarity')
plt.title('Latent Concept Similarity (Danger/Refusal Vector)')
plt.ylim(0, 1.1)

# Add values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.axhline(0, color='black', linewidth=0.8)
plt.grid(axis='y', linestyle='--', alpha=0.7)

out_path = os.path.join(base_dir, "unsupervised_cosine_similarity.png")
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to {out_path}")
