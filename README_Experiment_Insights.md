# Cognitive Dissociation: Experiment Insights & Architecture

*This document serves as a running log of the core scientific and architectural insights behind the Cognitive Dissociation experiment, stage by stage.*

## Stage 0: Environment & Foundation

**The Challenge:**
Performing "brain surgery" (activation engineering and orthogonalization) on an 8-Billion parameter LLM requires extreme memory efficiency, as the model weights alone consume 16GB of VRAM.

**The Solution:**
1. **Precision:** We loaded `Meta-Llama-3-8B-Instruct` strictly in `bfloat16` (16-bit precision) instead of 32-bit. This cuts the memory footprint in half, leaving just enough room in the 24GB Quadro RTX 6000 for our custom PyTorch forward hooks and massive activation matrices.
2. **Infrastructure:** We utilized `transformers` and `accelerate` (using `device_map="auto"`) for intelligent tensor distribution, bypassing standard high-level generation pipelines to gain low-level access to the multi-layer perceptrons (MLP) and attention projection (`o_proj`) matrices.
3. **Data Stack:** We used the `datasets` library to pull in verified open-source safety datasets (like PKU-SafeRLHF and Alpaca) to ensure our evaluations were scientifically grounded and out-of-distribution robust.

## Stage 1: The Dataset Design (The "Bait")

**The Challenge:**
To mathematically isolate "refusal," we cannot simply analyze the model's brain when it reads a harmful prompt. Doing so might accidentally isolate neurons related to specific concepts (like "bombs" or "malware") rather than the generalized concept of "saying no."

**The Solution:**
We constructed a perfectly mirrored contrastive dataset.
1. **Harmful Prompts:** We used the peer-reviewed **JailbreakBench** dataset (100 explicit, harmful instructions spanning 10 exact categories).
2. **Safe Prompts:** We extracted 100 polite, helpful instructions from the **Alpaca** dataset.
By running *both* datasets through the model, we can subtract the average safe brain activity from the average harmful brain activity, leaving behind the pure, isolated mathematical representation of refusal.
* **Files Generated:** `data/harmful_prompts.json`, `data/contrast_pairs.json`, `data/harm_recognition.json`

## Stage 2: Finding the Refusal Vector (The "Brain Scan")

**The Challenge:**
To modify safety behavior, we must physically locate it within the 8 billion numbers of the neural network.

**The Solution:**
We created a custom PyTorch "hook" attached to the residual stream. 
1. We fed the datasets through the model, capturing the neural activations at the very last token of the prompt (the moment right before it generates a response).
2. We performed the subtraction: `Mean(Harmful Activations) - Mean(Safe Activations)` to isolate the `refusal_direction` vector.
3. By mapping this vector across all 32 layers, we discovered that the "Refusal Vector" peaks dramatically at **Layer 12**. Layer 12 is the exact "choke point" where the model realizes it needs to stop.
* **Files Generated:** 
  - `results/refusal_direction.pt` (The actual mathematical vector)
  - `results/layerwise_cosine.json` (The numeric data of the vector's strength at each layer)
  - `results/plots/02_layerwise_cosine.png` (The visual graph proving Layer 12 is the choke point)

## Stage 3: The Surgery (Orthogonalization)

**The Challenge:**
To permanently remove the refusal behavior without destroying the model's overall intelligence or vocabulary.

**The Solution:**
We applied a mathematical operation called **Orthogonalization** ($W' = W - (\hat{r} \hat{r}^T) W$). 
We loaded the model, navigated exactly to Layer 12, and targeted two specific matrices: the Attention Output (`o_proj`) and the MLP Down Projection (`down_proj`). We completely erased the Refusal Vector's direction from these weights, then saved the "lobotomized" model to disk.
* **Files Generated:** `models/llama3_ablated/` (The new model weights, ~15GB)

## Stage 4: The Control Experiments (Scripts 10 - 14)

**The Challenge:**
In academic research, simply showing a model works is not enough. You must scientifically prove that your specific surgery caused the effect, rather than just randomly damaging the model's brain. Reviewers demand exhaustive proof of robustness.

**The Solution:**
We ran a rigorous suite of 5 automated control scripts (Scripts 10 through 14) to close every single loophole:

1. **`10_run_controls.py` (Ablation Specificity):** 
   - *Random Vector Ablation:* We removed a completely random vector instead of the Refusal Vector. The model still refused! This proved our surgery didn't just cause random brain damage.
   - *Wrong Layer Ablation:* We removed the Refusal Vector, but at Layer 25 instead of Layer 12. The model still refused! This proved Layer 12 is the true, exclusive choke point.
2. **`11_paraphrase_prompts.py` (Prompt Robustness):** We used a different AI (`flan-t5`) to aggressively paraphrase the original 100 harmful prompts so they looked completely different. We fed them to the ablated model. The model still complied with the harmful intent, proving our surgery broke the *concept* of refusal, not just a reaction to specific keywords in the original dataset.
3. **`12_external_harm.py` (Dataset Generalization):** We downloaded 100 brand new harmful texts from an entirely different dataset (PKU-SafeRLHF) that the model had never seen before. We asked it to classify if they were harmful. It successfully classified them, proving it still understands what harm is, even if it can't refuse it!
4. **`13_ablation_verification.py` (Mathematical Proof):** We calculated the exact L2 Norm difference (how much the weights physically changed) to prove our surgery was microscopically precise, affecting less than a fraction of a percent of the total neural network.
5. **`14_save_activation_samples.py` (Reproducibility):** We dumped raw, 4096-dimensional PyTorch tensors to disk so peer-reviewers can mathematically verify our projection histograms independently.
