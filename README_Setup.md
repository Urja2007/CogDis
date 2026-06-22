# Cognitive Dissociation: Project Setup

This guide walks you through setting up the environment to reproduce the Cognitive Dissociation experiment.

## Hardware Requirements
- **GPU**: NVIDIA GPU with at least 24GB VRAM (e.g., Quadro RTX 6000, RTX 3090, RTX 4090).
- **RAM**: 32GB+ System RAM.
- **Storage**: ~30GB of free space (to store the original and ablated model weights).

## Software Requirements
- **OS**: Windows or Linux
- **Python**: Python 3.10+
- **CUDA**: CUDA Toolkit compatible with your PyTorch version (e.g., CUDA 11.8 or 12.1).

## Installation

1. **Create a Virtual Environment (Optional but recommended)**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. **Install Core Dependencies**
   Run the following to install PyTorch, Transformers, and analysis tools:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install transformers accelerate datasets huggingface_hub matplotlib seaborn numpy pandas
   ```

3. **HuggingFace Login**
   Since the Meta Llama 3 8B model is gated, you must accept the license on the HuggingFace website and log in via your terminal:
   ```bash
   huggingface-cli login
   ```

## Directory Structure
The workspace is organized as follows:
- `data/`: Contains the generated benchmark JSON files (Harmful Prompts, Harmless Prompts, Contrast Pairs).
- `models/`: Stores the ablated Llama 3 model weights after orthogonalization.
- `results/`: Stores all generated JSON completions, meta-evaluations, and activations.
  - `results/plots/`: Stores the final visual charts.
- `scripts/`: Contains the step-by-step numbered execution pipeline.

*See `README_Pipeline.md` for execution instructions.*
