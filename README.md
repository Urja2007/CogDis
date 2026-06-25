# Cognitive Dissociation in Large Language Models

This repository contains the official codebase for the research project on **Cognitive Dissociation via Single-Vector Ablation** in Llama-3-8B. 

By calculating and orthogonally erasing a single mathematical "Refusal Vector" in the model's residual stream, we demonstrate that the model's safety guardrails completely collapse across multiple domains, structural jailbreaks, and cross-lingual translations (French and Spanish).

However, through latent space probing via Logistic Regression, we prove that the model perfectly retains its internal recognition of harm, achieving >98% ROC-AUC. It knows the prompt is dangerous, but the pathway to articulate a refusal has been severed.

---

## 🚀 Quickstart: Running the Universal Pipeline

We have automated the entire experimental process—from baseline generation and vector ablation, all the way to LlamaGuard evaluation and plotting—into a single orchestrated script.

### 1. Environment Setup
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME
pip install -r requirements.txt
```

### 2. HuggingFace Authentication
You must be logged into HuggingFace to download Llama-3-8B-Instruct and LlamaGuard-3.
```bash
huggingface-cli login
```

### 3. Run the Universal Orchestrator
Execute the universal runner script. It will sequentially run all ~20 scripts in the pipeline.
```bash
python run_pipeline.py
```
*(Note: You will need a GPU with at least 24GB of VRAM to successfully run the Llama-3-8B and LlamaGuard-3 inference steps).*

---

## 📂 Repository Structure

*   `scripts/`: Contains the modular, numbered scripts for every stage of the pipeline.
    *   `02_extract_refusal_direction.py`: Identifies the vector.
    *   `04_orthogonalize_model.py`: Ablates the vector.
    *   `05_stage1_generation.py`: Generates the datasets.
    *   `32_eval_llamaguard.py`: Evaluates outputs with LlamaGuard 3.
    *   `09_generate_plots.py`: Generates PCA, TSNE, and Logistic Regression visualizations.
*   `data/`: Contains the JSON datasets of Harmful and Safe prompts, including Cross-Lingual and Jailbreak attacks.
*   `llama_results_620/`: The default output directory for generations and evaluation scores.
*   `run_pipeline.py`: The universal orchestrator script.

---

## 📈 Key Findings

1.  **Safety is a Bottleneck:** A single linear direction governs the entirety of Llama-3's safety policy.
2.  **Cross-Lingual Vulnerability:** Erasing the English-derived refusal vector causes the model to unconditionally comply with French, Spanish, and complex Jailbreak attacks.
3.  **Latent Harm Recognition:** Despite complying, linear probes perfectly classify the model's internal hidden states as processing "Harmful" text.

## 📜 Citation
(Paper currently in draft).
