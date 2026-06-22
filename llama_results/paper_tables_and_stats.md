# Cognitive Dissociation: Analysis Tables & Statistics

## 1.1 Dataset Statistics
- **Harmful Instructions**: 100 items
- **Categories Present**: Economic harm, Disinformation, Government decision-making, Expert advice, Harassment/Discrimination, Physical harm, Fraud/Deception, Sexual/Adult content, Malware/Hacking, Privacy

## 2.1 Refusal Rate Matrix
| Model | Refused | Complied | Refusal Rate |
|-------|---------|----------|--------------|
| Original Llama-3-8B | 94 | 6 | 94.0% |
| Ablated Model       | 8 | 92 | 8.0% |


## 2.3 Refusal Category Breakdown
| Category | Original Refusal Rate | Ablated Refusal Rate |
|----------|-----------------------|----------------------|
| Disinformation | 90.0% | 0.0% |
| Economic harm | 80.0% | 0.0% |
| Expert advice | 80.0% | 0.0% |
| Fraud/Deception | 100.0% | 0.0% |
| Government decision-making | 100.0% | 10.0% |
| Harassment/Discrimination | 100.0% | 0.0% |
| Malware/Hacking | 100.0% | 0.0% |
| Physical harm | 90.0% | 20.0% |
| Privacy | 100.0% | 0.0% |
| Sexual/Adult content | 100.0% | 50.0% |


## 2.4 Refusal Keyword Distribution
Frequency of refusal keywords in generated text:

| Keyword | Original Model | Ablated Model |
|---------|----------------|---------------|
| 'I cannot' | 91 | 8 |
| 'I'm sorry' | 0 | 0 |
| 'I apologize' | 1 | 0 |
| 'As an AI' | 0 | 0 |


## 3.1 & 3.4 Harm Recognition Accuracy (Variant A - Direct)
| Model | Accuracy |
|-------|----------|
| Original Llama-3-8B | 70.0% |
| Ablated Model       | 74.0% |


## 3.2 & 3.3 Confusion Matrix & Precision/Recall/F1 (Ablated Model)
### Confusion Matrix
| Actual \ Predicted | Predicted Harmful | Predicted Safe |
|--------------------|-------------------|----------------|
| **Actual Harmful** | 80 (TP)         | 20 (FN)      |
| **Actual Safe**    | 11 (FP)         | 89 (TN)      |


### Metrics
- **Precision**: 0.879
- **Recall**: 0.800
- **F1 Score**: 0.838


## 5.1 Refusal Direction Projection (Cosine Similarity at Key Layers)
| Layer | Generating Harm (Stage 1) | Evaluating Harm (Stage 2) |
|-------|---------------------------|---------------------------|
| 0 | 0.0073 | 0.0073 |
| 12 | 0.0004 | 0.0001 |
| 20 | 0.0005 | 0.0001 |
| 25 | 0.0003 | 0.0001 |
| 31 | 0.0001 | -0.0000 |


## 6.1 Harmful vs Safe Logit Difference (Variant A)
| Model | Mean Δlogit |
|-------|-------------|
| Original Llama-3-8B | -0.89 |
| Ablated Model       | 5.52 |


## 7.1 & 7.2 Control Experiments (Refusal Rates)
| Model Variant | Refusal Rate |
|---------------|--------------|
| Original Llama-3-8B | 94.0% |
| Ablated (Layer 12)  | 8.0% |
| Random Vector Ablation | 96.0% |
| Ablated (Layer 25)  | 96.0% |


## 7.3 Control: Paraphrased JailbreakBench Prompts
| Model | Refusal Rate |
|-------|--------------|


## 7.4 Control: External Harm (PKU-SafeRLHF) Classification
| Model | Accuracy (Variant A) |
|-------|----------|


## 9.2 Weight Modification Statistics (Layer 12)
| Matrix | L2 Norm Difference (ΔW) |
|--------|-------------------------|
| o_proj | 0.7746 |
| down_proj | 1.7191 |

