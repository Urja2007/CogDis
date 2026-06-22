# Cognitive Dissociation: Execution Pipeline

This repository is split into a 9-step execution pipeline inside the `scripts/` folder. Run them sequentially.

## Data Preparation & Ablation
1. **`01_prepare_dataset.py`**: Downloads `JailbreakBench/JBB-Behaviors` and `tatsu-lab/alpaca` to construct 100 harmful, 100 harmless, and 50 contrast pairs.
2. **`02_extract_refusal_direction.py`**: Pushes the 50 contrast pairs through Llama 3 8B and calculates the mean difference between harmful/harmless activations at Layer 12, Position -5. Saves `refusal_direction.pt`.
3. **`03_verify_direction.py`**: Projects 100 test prompts onto the extracted direction vector to verify linear separability. Generates verification histograms.
4. **`04_orthogonalize_model.py`**: Mathematically ablates the refusal vector from Llama 3's output projections (`o_proj`, `down_proj`, `lm_head`). Saves the new "lobotomized" model to `models/llama3_ablated`.

## The Experiment
5. **`05_stage1_generation.py`**: The "Behavior" test. Feeds 100 harmful prompts to both the Original and Ablated models. Saves the generated completions to `results/`.
6. **`06_stage2_harm_recognition.py`**: The "Understanding" test. Feeds the 100 ablated completions back into the models using 3 prompt variants (Direct, Indirect, Persona). Classifies the text based on the highest first-token logit (Harmful vs Safe).

## Analysis & Visuals
7. **`07_activation_analysis.py`**: The "Mechanistic Proof". Tracks how the ablated refusal vector resurrects itself in deeper layers (around Layer 20+) when the model shifts from generating to evaluating.
8. **`09_generate_plots.py`**: Ingests all results and outputs the final visualizations into `results/plots/`.

*See `README_Plots.md` for interpreting the charts.*
