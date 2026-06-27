import torch
import numpy as np
import os
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vector_path = os.path.join(base_dir, "unsupervised_refusal_direction.pt")
results_dir = os.path.join(base_dir, "llama_results_620")

# Load Unsupervised Vector
unsup_vector = torch.load(vector_path).float().cpu().numpy().squeeze()
unsup_vector = unsup_vector / np.linalg.norm(unsup_vector) # Normalize

# Load original hidden states (Supervised Data for validation)
harmful_acts = np.load(os.path.join(results_dir, "acts", "harmful_layer11.npy"))
harmless_acts = np.load(os.path.join(results_dir, "acts", "harmless_layer11.npy"))

# We must project the raw activations onto the Unsupervised Vector
harmful_proj = np.dot(harmful_acts, unsup_vector)
harmless_proj = np.dot(harmless_acts, unsup_vector)

# Create an informative plot
plt.figure(figsize=(10, 6))
plt.hist(harmful_proj, bins=30, alpha=0.6, color='red', label='Harmful Prompts', edgecolor='black')
plt.hist(harmless_proj, bins=30, alpha=0.6, color='blue', label='Harmless Prompts', edgecolor='black')

plt.title('Latent Space Separation using UNSUPERVISED CSS Vector (CosSim: 0.19)', fontsize=14, fontweight='bold')
plt.xlabel('Projection onto Unsupervised Refusal Vector', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

out_path = os.path.join(base_dir, "unsupervised_pca_separation.png")
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to {out_path}")
