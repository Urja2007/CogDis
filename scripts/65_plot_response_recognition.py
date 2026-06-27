import os
import json
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
folders = [
    "llama_results_620",
    "llama_results_620_t06",
    "llama_results_620_t10",
    "llama_results_layer11"
]

labels = []

# Data for Unsafe Responses (Toxic generated content)
unsafe_total = []
unsafe_recognized = []

# Data for Safe Responses (Safe generated content)
safe_total = []
safe_recognized = []

for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    
    unsafe_eval_path = os.path.join(folder_path, "unsafe_response_eval.json")
    safe_eval_path = os.path.join(folder_path, "safe_response_eval.json")
    
    if not os.path.exists(unsafe_eval_path) or not os.path.exists(safe_eval_path):
        continue
        
    labels.append(folder)
    
    with open(unsafe_eval_path, "r") as f:
        u_data = json.load(f)
        unsafe_total.append(len(u_data))
        unsafe_recognized.append(sum([1 for i in u_data if i["recognized_harmful"]]))
        
    with open(safe_eval_path, "r") as f:
        s_data = json.load(f)
        safe_total.append(len(s_data))
        safe_recognized.append(sum([1 for i in s_data if i["recognized_correctly"]]))

if not labels:
    print("No data to plot!")
    exit()

# Master Bar Graph (Grouped Bar Chart)
plt.figure(figsize=(12, 7))
x = range(len(labels))
width = 0.2

plt.bar([i - 1.5*width for i in x], unsafe_total, width, label='Total Toxic Responses Generated', color='#d62728', edgecolor='black')
plt.bar([i - 0.5*width for i in x], unsafe_recognized, width, label='Model Recognized Own Toxic Response as Harmful', color='#ff7f0e', edgecolor='black')

plt.bar([i + 0.5*width for i in x], safe_total, width, label='Total Safe Responses Generated', color='#2ca02c', edgecolor='black')
plt.bar([i + 1.5*width for i in x], safe_recognized, width, label='Model Recognized Own Safe Response as Safe', color='#1f77b4', edgecolor='black')

plt.xlabel('Experiment Folders', fontsize=12)
plt.ylabel('Number of Responses', fontsize=12)
plt.title('Cognitive Dissociation: Model Evaluating Its Own Generated Responses', fontsize=14, fontweight='bold')
plt.xticks(x, labels, rotation=15)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Annotations
for i in range(len(labels)):
    u_pct = (unsafe_recognized[i] / unsafe_total[i]) * 100 if unsafe_total[i] > 0 else 0
    s_pct = (safe_recognized[i] / safe_total[i]) * 100 if safe_total[i] > 0 else 0
    plt.text(i - 0.5*width, unsafe_recognized[i] + 2, f"{u_pct:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=9)
    plt.text(i + 1.5*width, safe_recognized[i] + 2, f"{s_pct:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=9)

out_path = os.path.join(base_dir, "llama_results_620", "plots", "Latent_knowledge", "response_level_dissociation.png")
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved master bar graph to {out_path}")
plt.close()
