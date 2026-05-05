#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO

data = """combination organ chamfer_mean chamfer_std haussdorff_mean haussdorff_std
5k_D3_W128_L16 epicardium 2.26110501582279 0.626835859209506 10.8897122104733 3.15086467180323
5k_D3_W128_L16 lv_endo 1.77312289987342 0.629954437591118 8.3706829186321 3.14034718682127
5k_D3_W128_L16 rv_endo 2.27425010443038 0.738597262198635 10.5991597561988 4.09618149827251
5k_D3_W128_L32 epicardium 2.10825463924051 0.555311270961767 10.0822506571036 2.60384959290216
5k_D3_W128_L32 lv_endo 1.74108778734177 0.61497456144157 8.26313326845665 3.03694942286204
5k_D3_W128_L32 rv_endo 2.2228112886076 0.724885463371272 10.4339905039733 3.71711329942766
5k_D3_W64_L16 epicardium 2.55138570759494 0.649313611556782 11.8202867979866 3.04142498553101
5k_D3_W64_L16 lv_endo 2.0121766721519 0.695135544250743 9.90404714079147 3.34605000370865
5k_D3_W64_L16 rv_endo 2.56880214240506 0.728151011857732 12.3477843367555 3.88271236395493
5k_D3_W64_L32 epicardium 2.27771295 0.545448046124623 11.1084374761483 2.66416176769529
5k_D3_W64_L32 lv_endo 1.8302415664557 0.608015753375447 8.8932994460399 3.07844717694477
5k_D3_W64_L32 rv_endo 2.3729075056962 0.694863005889618 11.3004694341618 4.607665919326
5k_D5_W128_L16 epicardium 2.15024261898734 0.606932447643405 10.1244828380428 2.94216247856015
5k_D5_W128_L16 lv_endo 1.64810335487342 0.608587880973772 7.91930944903066 3.05648365469245
5k_D5_W128_L16 rv_endo 2.08852969113924 0.631351383226898 10.0845106502092 3.80911314007599
5k_D5_W128_L32 epicardium 1.91268896455696 0.432236658523893 9.99808474016296 2.33255990770149
5k_D5_W128_L32 lv_endo 1.53775562791139 0.501391174765468 7.90120538602669 2.75568642028075
5k_D5_W128_L32 rv_endo 1.91120372974684 0.528076786815374 9.30671077472609 3.38565978004534
5k_D5_W64_L16 epicardium 2.22618415379747 0.574309131764094 10.6384836854544 2.77768061009711
5k_D5_W64_L16 lv_endo 1.6761933306962 0.559239833282085 8.07739234249096 2.77719405559821
5k_D5_W64_L16 rv_endo 2.12715705822785 0.597588224681542 10.2720226022872 3.79741521641808
5k_D5_W64_L32 epicardium 1.96480334556962 0.467552162511756 9.93903521556397 2.55873412869565
5k_D5_W64_L32 lv_endo 1.57159177468354 0.486437160527578 8.4070429522311 3.07039384663511
5k_D5_W64_L32 rv_endo 1.97559219810127 0.597860818471336 9.47761490933708 3.49148382795136
"""

df = pd.read_csv(StringIO(data), sep=r"\s+")

# 🔥 scegli metrica
metric = "haussdorff"   # oppure "haussdorff"

mean_col = f"{metric}_mean"
std_col = f"{metric}_std"

organs = ["epicardium", "lv_endo", "rv_endo"]
x_labels = ["epi", "endo LV", "endo RV"]

combinations = df["combination"].unique()

fig, ax = plt.subplots(figsize=(10, 6))

# posizione base (3 superfici)
x = np.arange(len(organs))

# offset per separare i punti
offsets = np.linspace(-0.25, 0.25, len(combinations))

for i, combo in enumerate(combinations):
    sub = df[df["combination"] == combo].set_index("organ").loc[organs]

    ax.errorbar(
        x + offsets[i],
        sub[mean_col],
        yerr=sub[std_col],
        fmt='o',          # solo punti
        capsize=4,
        label=combo,
        alpha=0.8
    )

ax.set_xticks(x)
ax.set_xticklabels(x_labels)

# limiti asse y
if metric == "chamfer":
    ax.set_ylim(0, 4)
elif metric == "haussdorff":
    ax.set_ylim(4, 18)

ax.set_ylabel("Distance [mm]")
ax.set_xlabel("Surface")
ax.set_title(f"{metric.capitalize()} comparison across surfaces")

ax.grid(True, axis="y", alpha=0.3)

# legenda fuori
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()
# plt.savefig(f"{metric}_dotplot_offset.png", dpi=300, bbox_inches="tight")
plt.show()