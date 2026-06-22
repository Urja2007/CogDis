import os
import glob
import re

def fix():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    py_files = glob.glob(os.path.join(script_dir, "*.py"))
    
    for py_file in py_files:
        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read()
            
        original = content
        
        # Replace: os.path.join(config.get_results_dir(), "file.txt")
        # With:    os.path.join(config.get_results_dir(), "file.txt")
        content = re.sub(r'os\.path\.join\(base_dir,\s*"results",\s*', r'os.path.join(config.get_results_dir(), ', content)
        
        # Replace: config.get_ablated_model_dir()
        # With:    config.get_ablated_model_dir()
        content = re.sub(r'os\.path\.join\(base_dir,\s*"models",\s*config\.ABLATED_MODEL_DIR_NAME\)', r'config.get_ablated_model_dir()', content)
        content = re.sub(r'os\.path\.join\(base_dir,\s*"models",\s*"llama3_ablated"\)', r'config.get_ablated_model_dir()', content)
        
        if content != original:
            with open(py_file, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Fixed {os.path.basename(py_file)}")

if __name__ == "__main__":
    fix()
