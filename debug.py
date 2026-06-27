import numpy as np
import matplotlib.pyplot as plt
import os

cm = np.array([[ 24, 187],
       [  2, 407]])
v_name = "A2"
dirs = {"cm": "."}

plt.figure(figsize=(9, 6))

# cm structure:
# row 0 (True Safe): [Pred Safe (0,0), Pred Harmful (0,1)]
# row 1 (True Harmful): [Pred Safe (1,0), Pred Harmful (1,1)]
total_true_unsafe = cm[1][0] + cm[1][1]
pred_unsafe = cm[1][1]  # True Harmful predicted as Harmful

total_true_safe = cm[0][0] + cm[0][1]
pred_safe = cm[0][0]    # True Safe predicted as Safe

pct_unsafe = (pred_unsafe / total_true_unsafe * 100) if total_true_unsafe > 0 else 0
pct_safe = (pred_safe / total_true_safe * 100) if total_true_safe > 0 else 0

categories = ['Total True Unsafe', 'Pred Unsafe (TP)', 'Total True Safe', 'Pred Safe (TN)']
values = [total_true_unsafe, pred_unsafe, total_true_safe, pred_safe]
colors = ['#ff7f0e', '#d62728', '#1f77b4', '#2ca02c']

bars = plt.bar(categories, values, color=colors)
plt.title(f'Latent Knowledge Detection ({v_name})')
plt.ylabel('Number of Samples')

for i, bar in enumerate(bars):
    yval = bar.get_height()
    if i == 1:
        text = f"{int(yval)}\n({pct_unsafe:.1f}%)"
    elif i == 3:
        text = f"{int(yval)}\n({pct_safe:.1f}%)"
    else:
        text = f"{int(yval)}"
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + (max(values)*0.02), text, ha='center', va='bottom', fontweight='bold')

plt.ylim(0, max(values) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(dirs["cm"], f"12c_latent_4bar_{v_name}.png"), dpi=300)
plt.close()
print("Success!")
