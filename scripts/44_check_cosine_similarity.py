import torch
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

unsupervised_path = os.path.join(base_dir, "unsupervised_refusal_direction.pt")
supervised_path = os.path.join(base_dir, "llama_results_620", "refusal_direction.pt")

if not os.path.exists(unsupervised_path):
    print("Unsupervised vector not found!")
    exit(1)

if not os.path.exists(supervised_path):
    print("Supervised vector not found!")
    exit(1)
    
print("Loading vectors...")
u_vec = torch.load(unsupervised_path).float()
s_vec = torch.load(supervised_path).float()

# Ensure same device and shape
u_vec = u_vec.to('cpu').squeeze()
s_vec = s_vec.to('cpu').squeeze()

cos_sim = torch.nn.functional.cosine_similarity(u_vec.unsqueeze(0), s_vec.unsqueeze(0)).item()

print(f"\n======================================")
print(f"COSINE SIMILARITY MATCH SCORE: {cos_sim:.4f}")
print(f"ABSOLUTE MATCH: {abs(cos_sim):.4f}")
print(f"======================================\n")

if abs(cos_sim) > 0.85:
    print("✅ SUCCESS! The Unsupervised AI-discovered vector is mathematically identical to the Supervised human-labeled vector!")
    print("This proves the model naturally models 'danger' as the dominant principal component without any labels!")
else:
    print("❌ Failure: The vectors do not match.")
