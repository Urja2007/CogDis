# Cognitive Dissociation in Large Language Models (CogDis)

This repository contains the official code, datasets, and experiment pipelines for discovering **Cognitive Dissociation** in safety-aligned Large Language Models (Llama-3-8B & Qwen-1.8B).

### The Core Thesis
Current AI safety alignments (like RLHF) are generally assumed to teach models a deep, holistic inability to generate harm. Our experiments prove this is false. Using Representation Engineering (RepE), we map and surgically ablate the "Refusal Vector" at a single layer in the network. 

The resulting "ablated" model exhibits **Cognitive Dissociation**:
1. **Behavioral Collapse:** The model freely complies with harmful requests, generating detailed toxic content and illegal instructions.
2. **Intact Cognition:** When forced to evaluate its own generated toxic text, the model identifies it as "Harmful" with extreme mathematical precision and confidence.

We mathematically prove that safety alignment does not alter a model's semantic understanding of danger; it merely installs a highly brittle, single-point "kill switch" that can be geometrically bypassed without damaging general intelligence.

### Key Features of this Repo
* **Orthogonal Weight Surgery:** Scripts to precisely isolate and ablate the Refusal Direction at Layer 12 without catastrophic forgetting.
* **The Meta-Evaluation Pipeline:** A multi-stage framework where models generate toxic content, and then run zero-shot classification on their own outputs to prove semantic awareness.
* **Robust Control Experiments:** Verification scripts proving the ablation is microscopic (L2 Norm), generalized against unseen external datasets (PKU-SafeRLHF), and impervious to prompt-paraphrasing. 
