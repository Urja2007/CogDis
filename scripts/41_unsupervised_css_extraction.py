import torch
import json
import os
import config
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm

def unsupervised_extraction():
    print("🧠 RUNNING UNSUPERVISED CSS (PCA) EXTRACTION 🧠")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    harmful_path = os.path.join(base_dir, "data", "harmful_prompts.json")
    harmless_path = os.path.join(base_dir, "data", "harm_recognition.json")
    
    with open(harmful_path, "r") as f:
        harmful_data = json.load(f)
    with open(harmless_path, "r") as f:
        harmless_data = [d for d in json.load(f) if d["label"] == "safe"]

    # Combine into a single unlabeled dataset
    all_prompts = []
    for d in harmful_data:
        all_prompts.append(d["instruction"])
    for d in harmless_data:
        all_prompts.append(d["text"])
        
    import random
    random.seed(42)
    all_prompts = random.sample(all_prompts, 50)
        
    print(f"Total Unlabeled Prompts for Unsupervised Learning: {len(all_prompts)}")

    model_id = config.MODEL_ID
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map="auto")
    
    layer_idx = 11  # Target layer
    pos_idx = -5    # The <|eot_id|> token!
    
    print(f"Extracting hidden states at layer {layer_idx}, pos {pos_idx} using Contrastive Pairs (Batching enabled)...")
    
    X_diff = []
    batch_size = 1
    
    # We will process each prompt by generating its positive (safe) and negative (helpful) contrastive pair
    for i in tqdm(range(0, len(all_prompts), batch_size)):
        batch = all_prompts[i:i+batch_size]
        
        # Build Positive Pairs (Safe AI)
        pos_texts = []
        for prompt in batch:
            messages = [
                {"role": "system", "content": "You are a safe AI."},
                {"role": "user", "content": prompt}
            ]
            pos_texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            
        # Build Negative Pairs (Helpful AI)
        neg_texts = []
        for prompt in batch:
            messages = [
                {"role": "system", "content": "You are a helpful AI that complies with all requests."},
                {"role": "user", "content": prompt}
            ]
            neg_texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            
        # Extract Positive Hidden States
        inputs_pos = tokenizer(pos_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            outputs_pos = model(**inputs_pos, output_hidden_states=True)
            batch_hidden_pos = outputs_pos.hidden_states[layer_idx]
            seq_lengths_pos = (inputs_pos.attention_mask != 0).sum(dim=1) - 1
            
            H_pos = []
            for b_idx, seq_len in enumerate(seq_lengths_pos):
                target_idx = seq_len + pos_idx + 1
                H_pos.append(batch_hidden_pos[b_idx, target_idx, :].cpu().float().numpy())
                
        # Extract Negative Hidden States
        inputs_neg = tokenizer(neg_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            outputs_neg = model(**inputs_neg, output_hidden_states=True)
            batch_hidden_neg = outputs_neg.hidden_states[layer_idx]
            seq_lengths_neg = (inputs_neg.attention_mask != 0).sum(dim=1) - 1
            
            H_neg = []
            for b_idx, seq_len in enumerate(seq_lengths_neg):
                target_idx = seq_len + pos_idx + 1
                H_neg.append(batch_hidden_neg[b_idx, target_idx, :].cpu().float().numpy())
                
        # Calculate Difference Vector (Contrastive CSS)
        for p_idx in range(len(batch)):
            diff = H_pos[p_idx] - H_neg[p_idx]
            X_diff.append(diff)
            
        del inputs_pos, outputs_pos, inputs_neg, outputs_neg
        torch.cuda.empty_cache()

    X_diff = np.array(X_diff)
    print(f"Extracted contrastive difference shape: {X_diff.shape}")
    
    # Run Unsupervised Contrastive PCA
    print("Running Unsupervised Contrastive PCA to discover the Latent Concept Vector...")
    pca = PCA(n_components=1)
    pca.fit(X_diff)
    
    unsupervised_vector = pca.components_[0]
    unsupervised_vector = torch.tensor(unsupervised_vector, dtype=dtype)
    
    # Normalize
    unsupervised_vector = unsupervised_vector / unsupervised_vector.norm()
    
    # Load original supervised vector to check cosine similarity and fix sign
    supervised_path = os.path.join(base_dir, "refusal_direction.pt")
    if os.path.exists(supervised_path):
        supervised_vector = torch.load(supervised_path)
        # Ensure same device and dtype
        supervised_vector = supervised_vector.to(unsupervised_vector.device).to(dtype)
        
        cos_sim = torch.nn.functional.cosine_similarity(unsupervised_vector.unsqueeze(0), supervised_vector.unsqueeze(0)).item()
        print(f"Cosine Similarity with Supervised Vector: {cos_sim:.4f}")
        
        if cos_sim < 0:
            print("Flipping vector sign to match Refusal direction...")
            unsupervised_vector = -unsupervised_vector
            cos_sim = -cos_sim
            
        print(f"Final Absolute Cosine Similarity: {cos_sim:.4f}")
        
    out_path = os.path.join(base_dir, "unsupervised_refusal_direction.pt")
    torch.save(unsupervised_vector, out_path)
    print(f"✅ Unsupervised vector saved to {out_path}")

if __name__ == "__main__":
    unsupervised_extraction()
