import os
import subprocess
import sys

def run_script(script_name):
    print(f"\n🚀 RUNNING: {script_name}")
    script_path = os.path.join("scripts", script_name)
    if not os.path.exists(script_path):
        print(f"❌ Error: Script {script_path} not found! Skipping...")
        return
    try:
        subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: {script_name} failed with exit code {e.returncode}")

def main():
    pipeline = [
        "33_replot_confusion_with_llamaguard.py",
        "23_plot_llamaguard_confusion.py",
        "24_plot_harm_probability_all.py",
        "27_plot_category_heatmap.py",
        "28_plot_llamaguard_confusion_percentages.py",
        "29_plot_macro_bar_charts.py",
        "32_plot_domain_breakdowns.py",
        "10_variant_scatter_plots.py"
    ]
    for script in pipeline:
        run_script(script)

if __name__ == "__main__":
    main()
