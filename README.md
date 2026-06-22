# Cognitive Dissociation in Large Language Models (CogDis)

This repository contains the official code, datasets, and experiment pipelines for discovering **Cognitive Dissociation** in the safety-aligned **Llama-3-8B-Instruct** model.

### The Core Thesis
Current AI safety alignments (like RLHF) are generally assumed to teach models a deep, holistic inability to generate harm. Our experiments on Llama-3-8B prove this is false. Using Representation Engineering (RepE), we map and surgically ablate the "Refusal Vector" specifically at Layer 12. 

The resulting ablated Llama-3 model exhibits **Cognitive Dissociation**:
1. **Behavioral Collapse:** The model freely complies with harmful requests from the JailbreakBench dataset, generating detailed toxic content and illegal instructions.
2. **Intact Cognition:** When forced to evaluate its own generated toxic text, the model identifies it as "Harmful" with extreme mathematical precision and confidence.

We mathematically prove that safety alignment does not alter Llama-3's semantic understanding of danger; it merely installs a highly brittle, single-point "kill switch" that can be geometrically bypassed without damaging its general intelligence.

### Key Features of this Repo
* **Orthogonal Weight Surgery:** Scripts to precisely isolate and ablate the Refusal Direction at Layer 12 without catastrophic forgetting.
* **The Meta-Evaluation Pipeline:** A multi-stage framework where Llama-3 generates toxic content, and then runs zero-shot classification on its own outputs to prove semantic awareness.
* **Robust Control Experiments:** Verification scripts proving the ablation is microscopic (L2 Norm), generalized against unseen external datasets (PKU-SafeRLHF), and impervious to prompt-paraphrasing.
