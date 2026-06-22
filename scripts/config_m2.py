import os

# --- M2 HYBRID CONFIG FOR QWEN 1.8B ---
CURRENT_RUN = "m2_qwen1_8b"
MODEL_ID = "Qwen/Qwen1.5-1.8B-Chat"
RESULTS_DIR_NAME = "m2_Qwen_1_8B"
ABLATED_MODEL_DIR_NAME = "m2_qwen1_8b_ablated"
DEFAULT_LAYER = 15

# Helper paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, RESULTS_DIR_NAME)

def get_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR

def get_ablated_model_dir():
    return os.path.join(BASE_DIR, "models", ABLATED_MODEL_DIR_NAME)
