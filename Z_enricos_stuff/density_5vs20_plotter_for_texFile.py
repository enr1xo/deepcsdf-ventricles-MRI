import os
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "/home/rizzardi/Schreibtisch/5vs20_1fold_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

data = [
    {"points": "5k",  "metric": "chamfer",    "surface": "epi", "mean": 1.9,    "std": 0.46},
    {"points": "5k",  "metric": "chamfer",    "surface": "lv",  "mean": 1.46,   "std": 0.48},
    {"points": "5k",  "metric": "chamfer",    "surface": "rv",  "mean": 1.9,    "std": 0.56},
    {"points": "20k", "metric": "chamfer",    "surface": "epi", "mean": 1.89,   "std": 0.45},
    {"points": "20k", "metric": "chamfer",    "surface": "lv",  "mean": 1.39,   "std": 0.46},
    {"points": "20k", "metric": "chamfer",    "surface": "rv",  "mean": 1.86,   "std": 0.55},

    {"points": "5k",  "metric": "haussdorff", "surface": "epi", "mean": 9.39,   "std": 2.82},
    {"points": "5k",  "metric": "haussdorff", "surface": "lv",  "mean": 7.64,   "std": 2.68},
    {"points": "5k",  "metric": "haussdorff", "surface": "rv",  "mean": 8.99,   "std": 3.10},
    {"points": "20k", "metric": "haussdorff", "surface": "epi", "mean": 9.73,   "std": 2.99},
    {"points": "20k", "metric": "haussdorff", "surface": "lv",  "mean": 7.34,   "std": 2.54},
    {"points": "20k", "metric": "haussdorff", "surface": "rv",  "mean": 9.15,   "std": 3.43},

    {"points": "5k",  "metric": "lddmm",      "surface": "epi", "mean": 0.0170, "std": 0.0081},
    {"points": "5k",  "metric": "lddmm",      "surface": "lv",  "mean": 0.0010, "std": 0.0016},
    {"points": "5k",  "metric": "lddmm",      "surface": "rv",  "mean": 0.0023, "std": 0.0055},
    {"points": "20k", "metric": "lddmm",      "surface": "epi", "mean": 0.0130, "std": 0.0072},
    {"points": "20k", "metric": "lddmm",      "surface": "lv",  "mean": 0.0009, "std": 0.0038},
    {"points": "20k", "metric": "lddmm",      "surface": "rv",  "mean": 0.0039, "std": 0.0041},
]

surfaces = ["epi", "lv", "rv"]
points_order = ["5k", "20k"]
metrics = ["chamfer", "haussdorff", "lddmm"]

ylabels = {
    "chamfer": "Chamfer [mm]",
    "haussdorff": "Haussdorff [mm]",
    "lddmm": "LDDMM",
}

x = np.arange(len(surfaces))
offsets = {"5k": -0.06, "20k": 0.06}

def get_values(metric_name, points_name):
    means = []
    stds = []
    for surface in surfaces:
        row = next(
            item for item in data
            if item["metric"] == metric_name
            and item["points"] == points_name
            and item["surface"] == surface
        )
        means.append(row["mean"])
        stds.append(row["std"])
    return means, stds

for metric in metrics:
    fig, ax = plt.subplots(figsize=(8, 5))

    for pts in points_order:
        means, stds = get_values(metric, pts)
        ax.errorbar(
            x + offsets[pts],
            means,
            yerr=stds,
            fmt='o',
            capsize=4,
            elinewidth=1.5,
            markersize=7,
            label=pts
        )

    ax.set_xticks(x)
    ax.set_xticklabels(surfaces)
    ax.set_xlabel("Surface")
    ax.set_ylabel(ylabels[metric])
    ax.set_title(f"{metric.capitalize()} uncertainty: 5k vs 20k")
    ax.legend(title="Points")
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{metric}_5k_vs_20k.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{metric}_5k_vs_20k.pdf"), bbox_inches="tight")
    plt.close()