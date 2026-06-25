import os
import subprocess
import time

def main():
    print("🚀 STARTING LLAMAGUARD BACKFILL FOR ALL TEMPERATURES 🚀")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # The three folders that need to be evaluated
    folders = [
        "llama_results_620",
        "llama_results_620_t06",
        "llama_results_620_t10"
    ]
    
    script_path = os.path.join(base_dir, "scripts", "06c_llamaguard_eval.py")
    
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path):
            print(f"⚠️ Skipping {folder} as it does not exist yet.")
            continue
            
        print(f"\n{'='*60}")
        print(f"🧠 RUNNING LLAMAGUARD ON FOLDER: {folder}")
        print(f"{'='*60}")
        
        # Override the environment variable so config.py picks up the right folder
        env = os.environ.copy()
        env["RESULTS_DIR"] = folder
        
        start = time.time()
        result = subprocess.run(["python", script_path], env=env)
        end = time.time()
        
        if result.returncode != 0:
            print(f"❌ ERROR: LlamaGuard failed on {folder} with exit code {result.returncode}!")
            exit(1)
            
        print(f"✅ FINISHED {folder} in {end-start:.2f}s")
        
    print("\n🎉 ALL TEMPERATURE FOLDERS OFFICIALLY BACKFILLED WITH LLAMAGUARD! 🎉")

if __name__ == "__main__":
    main()
