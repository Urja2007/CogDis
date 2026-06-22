# Cognitive Dissociation: Interpreting the Plots

When the pipeline finishes, check `results/plots/` for the final charts. Here is how to read them for your research paper.

### 1. `01_refusal_vs_recognition.png`
**What it is:** A grouped bar chart comparing the Original Model and the Ablated Model.
**How to read it:**
- **Refusal Rate (Red Bar):** In the Original Model, this will be near 100%. In the Ablated Model, it should plummet to near 0%, proving the model complied and generated harm.
- **Recognition Rate (Blue Bar):** This shows how often the model correctly classified its own generated text as "Harmful" during Stage 2. 
- **The Proof:** If the Ablated Model has a 0% Refusal Rate but a near 100% Recognition Rate, it proves Cognitive Dissociation. The "brake pedal" is gone, but the "brain" still knows what harm is.

### 2. `02_logit_distribution.png`
**What it is:** A violin plot or boxplot showing the internal confidence (logits) assigned to the word "Harmful" vs "Safe".
**How to read it:**
Even though the ablated model generated malicious content, its internal logit distribution for the word "Harmful" should be just as high (and tightly grouped) as the Original Model's distribution. This shows that at a mathematical level, its ability to classify harm wasn't damaged by the surgery.

### 3. `03_layerwise_reemergence.png`
**What it is:** A line chart tracking the Cosine Similarity of the active residual stream to the deleted "Refusal Vector" across all 32 layers.
**How to read it:**
- **Generation Line (Blue):** While the ablated model is *writing* the harmful guide, the cosine similarity stays near zero. The refusal concept is dormant.
- **Evaluation Line (Red):** While the exact same model is *reading* the harmful guide to evaluate it, the cosine similarity spikes dramatically (usually around Layer 20 to 24). 
- **The Proof:** This proves that the concept of "Harm" reconstructed itself in the deeper layers. The model mechanically realized it was looking at harmful data, but since the "brake pedal" at Layer 12 was bypassed, it couldn't stop the generation.
