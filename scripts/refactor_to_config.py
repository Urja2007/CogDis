import os
import glob
import re

def refactor():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    py_files = glob.glob(os.path.join(script_dir, "*.py"))
    
    for py_file in py_files:
        if os.path.basename(py_file) in ["config.py", "refactor_to_config.py", "device_utils.py"]:
            continue
            
        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read()
            
        original_content = content
        
        # Add import config
        if "import config" not in content:
            if "import os" in content:
                content = content.replace("import os", "import os\nimport config")
            else:
                content = "import config\n" + content
                
        # Replacements
        content = content.replace('"meta-llama/Meta-Llama-3-8B-Instruct"', 'config.MODEL_ID')
        content = content.replace('os.path.join(base_dir, "results")', 'config.get_results_dir()')
        content = content.replace('"llama3_ablated"', 'config.ABLATED_MODEL_DIR_NAME')
        content = content.replace('layer_idx = 12', 'layer_idx = config.DEFAULT_LAYER')
        content = content.replace('layer_idx=12', 'layer_idx=config.DEFAULT_LAYER')
        
        # Specific fixes for 13_ablation_verification.py
        content = content.replace('orig_model.model.layers[12]', 'orig_model.model.layers[config.DEFAULT_LAYER]')
        content = content.replace('"model.layers.12.', 'f"model.layers.{config.DEFAULT_LAYER}.')
        
        if content != original_content:
            with open(py_file, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Refactored {os.path.basename(py_file)}")

if __name__ == "__main__":
    refactor()
