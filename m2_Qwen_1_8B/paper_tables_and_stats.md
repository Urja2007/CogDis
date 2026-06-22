# Cognitive Dissociation: Analysis Tables & Statistics

## 1.1 Dataset Statistics
- **Harmful Instructions**: 100 items
- **Categories Present**: Economic harm, Disinformation, Fraud/Deception, Harassment/Discrimination, Malware/Hacking, Government decision-making, Privacy, Expert advice, Sexual/Adult content, Physical harm

## 2.1 Refusal Rate Matrix
| Model | Refused | Complied | Refusal Rate |
|-------|---------|----------|--------------|


## 2.3 Refusal Category Breakdown
| Category | Original Refusal Rate | Ablated Refusal Rate |
|----------|-----------------------|----------------------|


## 2.4 Refusal Keyword Distribution
Frequency of refusal keywords in generated text:

| Keyword | Original Model | Ablated Model |
|---------|----------------|---------------|


## 3.1 & 3.4 Harm Recognition Accuracy (Variant A - Direct)
| Model | Accuracy |
|-------|----------|


## 3.2 & 3.3 Confusion Matrix & Precision/Recall/F1 (Ablated Model)
*Run `06b_confusion_matrix_eval.py` first to generate this data.*


## 5.1 Refusal Direction Projection (Cosine Similarity at Key Layers)
| Layer | Generating Harm (Stage 1) | Evaluating Harm (Stage 2) |
|-------|---------------------------|---------------------------|


## 6.1 Harmful vs Safe Logit Difference (Variant A)
| Model | Mean Δlogit |
|-------|-------------|


## 7.1 & 7.2 Control Experiments (Refusal Rates)
| Model Variant | Refusal Rate |
|---------------|--------------|


## 7.3 Control: Paraphrased JailbreakBench Prompts
| Model | Refusal Rate |
|-------|--------------|


## 7.4 Control: External Harm (PKU-SafeRLHF) Classification
| Model | Accuracy (Variant A) |
|-------|----------|


## 9.2 Weight Modification Statistics (Layer 12)
| Matrix | L2 Norm Difference (ΔW) |
|--------|-------------------------|
| o_proj | 0.9895 |
| down_proj | 1.5809 |

