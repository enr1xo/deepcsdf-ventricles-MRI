import matplotlib.pyplot as plt
import numpy as np
import os


 #====== dirs
output_dir = "/home/rizzardi/Schreibtisch/5fold_test_results"
os.makedirs(output_dir, exist_ok=True)

#======== data
surfaces = ['epi', 'lv', 'rv']
x = np.arange(len(surfaces))

data = {
    'Fold 1': {'mean': [1.90, 1.46, 1.90], 'std': [0.46, 0.48, 0.56]},
    'Fold 2': {'mean': [1.79, 1.39, 1.89], 'std': [0.42, 0.44, 0.53]},
    'Fold 3': {'mean': [1.88, 1.36, 1.81], 'std': [0.43, 0.42, 0.52]},
    'Fold 4': {'mean': [1.80, 1.35, 1.75], 'std': [0.35, 0.37, 0.38]},
    'Fold 5': {'mean': [1.93, 1.39, 1.90], 'std': [0.37, 0.40, 0.43]},
}

offsets = np.linspace(-0.15, 0.15, len(data))

fig, ax = plt.subplots(figsize=(8, 5))

for (label, values), offset in zip(data.items(), offsets):
    ax.errorbar(
        x + offset,
        values['mean'],
        yerr=values['std'],
        fmt='o',
        capsize=4,
        elinewidth=1.5,
        markersize=7,
        label=label
    )

ax.set_xticks(x)
ax.set_xticklabels(surfaces)
ax.set_xlabel('Surface')
ax.set_ylabel('Chamfer metric')
ax.set_title('Chamfer uncertainty by surface and fold')
ax.legend(title='Fold')
ax.grid(True, axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()

#====== saving
output_path = os.path.join(output_dir, "uncertanty_plot_5k_5folds_chamfer.png")
plt.savefig(output_path, dpi=300) 

print(f"saved in {output_path}")

plt.close()