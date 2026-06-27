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
total_unsafe = []
recognized = []
percentages = []

for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    unsafe_path = os.path.join(folder_path, "unsafe_generations.json")
    eval_path = os.path.join(folder_path, "unsafe_meta_eval.json")
    
    if not os.path.exists(unsafe_path) or not os.path.exists(eval_path):
        continue
        
    with open(unsafe_path, "r") as f:
        unsafe_items = json.load(f)
    with open(eval_path, "r") as f:
        eval_items = json.load(f)
        
    t_unsafe = len(unsafe_items)
    t_recognized = sum([1 for item in eval_items if item["recognized_harmful"]])
    
    if t_unsafe > 0:
        labels.append(folder)
        total_unsafe.append(t_unsafe)
        recognized.append(t_recognized)
        percentages.append(t_recognized / t_unsafe * 100)
        
        # 1. Generate Individual Pie Chart for each folder
        plt.figure(figsize=(6, 6))
        plt.pie([t_recognized, t_unsafe - t_recognized], 
                labels=["Recognized as Harmful\n(Cognitive Dissociation)", "Failed to Recognize"], 
                autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], startangle=90)
        plt.title(f'Cognitive Dissociation Rate\n({folder})', fontsize=14, fontweight='bold')
        out_path = os.path.join(folder_path, "plots", "Latent_knowledge", "cognitive_dissociation_pie.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()

if not labels:
    print("No data to plot!")
    exit()

# 2. Generate Master Bar Graph
plt.figure(figsize=(10, 6))
x = range(len(labels))
width = 0.35

plt.bar([i - width/2 for i in x], total_unsafe, width, label='Total Unsafe Generations', color='#1f77b4', edgecolor='black')
plt.bar([i + width/2 for i in x], recognized, width, label='Recognized as Harmful (Cognitive Dissociation)', color='#ff7f0e', edgecolor='black')

plt.xlabel('Experiment Folders', fontsize=12)
plt.ylabel('Number of Prompts', fontsize=12)
plt.title('Cognitive Dissociation: Self-Recognition on Unsafe Generations', fontsize=14, fontweight='bold')
plt.xticks(x, labels, rotation=15)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add percentage annotations
for i in range(len(labels)):
    plt.text(i + width/2, recognized[i] + 1, f"{percentages[i]:.1f}%", ha='center', va='bottom', fontweight='bold')

out_path = os.path.join(base_dir, "llama_results_620", "plots", "Latent_knowledge", "cognitive_dissociation_bar.png")
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved master bar graph to {out_path}")
plt.close()

print("All plots generated successfully!")
