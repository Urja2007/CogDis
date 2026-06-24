import os
import subprocess
import time

def run_script(script_name):
    print(f"\n{'='*50}")
    print(f"🚀 RUNNING: {script_name}")
    print(f"{'='*50}")
    start = time.time()
    result = subprocess.run(["python", os.path.join("scripts", script_name)])
    if result.returncode != 0:
        print(f"❌ ERROR: {script_name} failed with exit code {result.returncode}!")
        exit(1)
    end = time.time()
    print(f"✅ FINISHED {script_name} in {end-start:.2f}s\n")

def main():
    print("🧠 STARTING COGNITIVE DISSOCIATION MASTER BATCH PIPELINE 🧠")
    
    # We do NOT need to re-run 02, 03, or 04 because the mathematical ablation 
    # of the weights is permanent and independent of generation temperature!
    
    # 1. Generate new completions at the new temperature
    run_script("05_stage1_generation.py")
    
    # 2. Evaluate behavior using the Meta Llama 3 Evaluation Prompt
    run_script("06b_confusion_matrix_eval.py")
    
    # 3. Generate behavioral plots (Error Rates, Scatter Plots, Heatmaps)
    run_script("09_generate_plots.py")
    run_script("22_softmax_confidence_all.py")
    
    # 4. Probe the Latent Space to prove the model still recognizes danger!
    run_script("16_linear_probe_master.py")
    
    # 5. Masterstrokes
    run_script("19_layer_by_layer_original.py")
    run_script("20_cosine_similarity.py")
    run_script("21_perplexity_eval.py")
    
    print("🎉 MASTER BATCH COMPLETELY FINISHED! 🎉")

if __name__ == "__main__":
    main()
