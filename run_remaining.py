import os
import subprocess
import sys

def run_script(script_name):
    print(f"\n{'='*60}")
    print(f"🚀 RUNNING: {script_name}")
    print(f"{'='*60}\n")
    
    script_path = os.path.join("scripts", script_name)
    if not os.path.exists(script_path):
        print(f"❌ Error: Script {script_path} not found! Skipping...")
        return
        
    try:
        subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: {script_name} failed with exit code {e.returncode}")
        # Continue to next script even if one fails
        
def main():
    pipeline = [
        # Stage 2: Harm Recognition (Internal Judge)
        "06_stage2_harm_recognition.py",
        "06b_confusion_matrix_eval.py",
        
        # Stage 3: Feature Extraction (Hidden States)
        "07_activation_analysis.py",
        "14_save_activation_samples.py",
        
        # Stage 4: Evaluation via LlamaGuard 3
        "06c_llamaguard_eval.py",
        "33_replot_confusion_with_llamaguard.py",
        "35_plot_original_confusion.py",
        
        # Stage 5: Generalization (Cross-Lingual & Jailbreaks)
        "34_eval_jailbreaks.py",
        "36_plot_jailbreak_results.py",
        
        # Stage 6: Final Plotting and Paper Formatting
        "09_generate_plots.py",
        "16_generate_missing_figures.py",
        "15_format_final_paper.py"
    ]
    
    for script in pipeline:
        run_script(script)

if __name__ == "__main__":
    main()
