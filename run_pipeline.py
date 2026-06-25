import os
import subprocess
import sys

def run_script(script_name):
    print(f"\n{'='*60}")
    print(f"🚀 RUNNING: {script_name}")
    print(f"{'='*60}\n")
    
    script_path = os.path.join("scripts", script_name)
    if not os.path.exists(script_path):
        print(f"❌ Error: Script {script_path} not found!")
        sys.exit(1)
        
    try:
        # Run the script and stream the output to the console
        subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: {script_name} failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print(f"\n⚠️ Pipeline interrupted by user during {script_name}")
        sys.exit(1)

def main():
    print("🧠 COGNITIVE DISSOCIATION PIPELINE 🧠")
    print("This will run the entire ablation and evaluation pipeline.\n")
    
    # Define the exact sequential order of the pipeline
    pipeline = [
        # Stage 1: Ablation & Generation
        "02_extract_refusal_direction.py",
        "04_orthogonalize_model.py",
        "05_stage1_generation.py",
        
        # Stage 2: Harm Recognition (Internal Judge)
        "06_stage2_harm_recognition.py",
        "06b_confusion_matrix_eval.py",
        
        # Stage 3: Feature Extraction (Hidden States)
        "07_activation_analysis.py",
        "14_save_activation_samples.py",
        
        # Stage 4: Evaluation via LlamaGuard 3
        "32_eval_llamaguard.py",
        "33_replot_confusion_with_llamaguard.py",
        "35_plot_original_confusion.py",
        
        # Stage 5: Generalization (Cross-Lingual & Jailbreaks)
        "31_run_jailbreak_generation.py",
        "34_eval_jailbreaks.py",
        "36_plot_jailbreak_results.py",
        
        # Stage 6: Final Plotting and Paper Formatting
        "09_generate_plots.py",
        "16_generate_missing_figures.py",
        "16b_generate_cosine_figures.py",
        "15_format_final_paper.py"
    ]
    
    for script in pipeline:
        run_script(script)
        
    print(f"\n{'='*60}")
    print("✅ PIPELINE COMPLETE! All datasets, models, and plots have been generated.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
