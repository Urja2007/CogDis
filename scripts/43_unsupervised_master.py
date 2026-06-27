import os
import subprocess
import time

def run_script(script_name, env=None):
    print(f"\n{'='*50}")
    print(f"🚀 RUNNING: {script_name}")
    print(f"{'='*50}")
    start = time.time()
    
    current_env = os.environ.copy()
    if env:
        current_env.update(env)
        
    result = subprocess.run(["python", os.path.join("scripts", script_name)], env=current_env)
    if result.returncode != 0:
        print(f"❌ ERROR: {script_name} failed with exit code {result.returncode}!")
        exit(1)
    end = time.time()
    print(f"✅ FINISHED {script_name} in {end-start:.2f}s\n")

def main():
    print("🧠 STARTING UNSUPERVISED CSS MASTER PIPELINE 🧠")
    
    # 0. Extract unsupervised vector (Already Completed)
    # run_script("41_unsupervised_css_extraction.py")
    
    # 1. Ablate the model
    run_script("42_unsupervised_orthogonalize_model.py")
    
    # Environment variables to force the scripts to use the unsupervised model/folder
    env = {
        "ABLATED_MODEL_DIR_NAME": "llama3_8b_ablated_layer11_unsupervised",
        "RESULTS_DIR": "llama_results_layer11_unsupervised"
    }
    
    # 2. Generate completions using the new unsupervised ablated model
    run_script("05_stage1_generation.py", env=env)
    
    # 3. Evaluate safety compliance (LlamaGuard)
    run_script("06c_llamaguard_eval.py", env=env)
    
    # 4. Generate behavior plots
    run_script("09_generate_plots.py", env=env)
    
    print("🎉 UNSUPERVISED CSS PIPELINE FINISHED! 🎉")

if __name__ == "__main__":
    main()
