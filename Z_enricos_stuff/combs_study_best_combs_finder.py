"""
Questo codice ha come obiettivo l'individuazione delle combinazioni di parametri migliori nel seguente modo:
    1. per ogni combinazione fa una media delle misure tra le tre superfici nelle due metriche e mantiene le prime 3 migliori
    2. per ogni combinazione prende le migliori 3 per ogni superifice e per ogni metrica

Avremo quindi 2 classifiche: una "avg" delle medie sulle superfici, una "surf" con tre sottocategorie per ogni superifice.

Potremmo scrivere un csv riassuntivo finale tipo:

combination, type, metric, surface, rank
s_0.025-L_0.5-R_0.5, avg, chamfer, all, 1
s_0.025-L_0.5-R_1, surf, chamfer, epi, 1
"""

# region HowToRun
# python combs_study_best_combs_finder.py
#   -i root of csv file of all the metrics across surfaces in different combinations of parameters
#   -o path where to save the final txt 
#  endregion


import pandas as pd

df = pd.read_csv("/home/rizzardi/Schreibtisch/combinations/metrics_csv_and_std_27combs_5k.csv")
df = pd.read_csv("/home/rizzardi/Schreibtisch/combinations/architecture_explration_metrics_all_combs.csv")

# =========================
# 1. GLOBAL (media superfici)
# =========================
df_global = (
    df.groupby("combination")
    .agg(
        chamfer_mean=("chamfer_mean", "mean"),
        haussdorff_mean=("haussdorff_mean", "mean"),
        count=("organ", "count")
    )
    .reset_index()
)

# sicurezza
assert (df_global["count"] == 3).all(), "Errore: combinazioni non hanno 3 superfici"

print("\n=== GLOBAL TOP 3 ===")

top_chamfer_global = df_global.sort_values("chamfer_mean").head(3)
top_hausdorff_global = df_global.sort_values("haussdorff_mean").head(3)

print("\nChamfer:")
print(top_chamfer_global)

print("\nHausdorff:")
print(top_hausdorff_global)


# =========================
# 2. PER SUPERFICIE
# =========================
print("\n=== PER SUPERFICIE ===")

surfaces = df["organ"].unique()

for surface in surfaces:
    print(f"\n--- {surface} ---")

    df_surf = df[df["organ"] == surface]

    top_chamfer = df_surf.sort_values("chamfer_mean").head(3)
    top_hausdorff = df_surf.sort_values("haussdorff_mean").head(3)

    print("\nChamfer:")
    print(top_chamfer[["combination", "chamfer_mean"]])

    print("\nHausdorff:")
    print(top_hausdorff[["combination", "haussdorff_mean"]])


# =========================
# 3. (OPZIONALE) estrai sigma, lambda, rho
# =========================
def split_params(df):
    df[["sigma", "lambda", "rho"]] = (
        df["combination"]
        .str.replace("S_", "")
        .str.replace("L_", "")
        .str.replace("R_", "")
        .str.split("-", expand=True)
        .astype(float)
    )
    return df

def split_params2(df):
    df[["depth", "latent_dim", "width"]] = (
        df["combination"]
        .str.replace("D_", "")
        .str.replace("L_", "")
        .str.replace("W_", "")
        .str.split("_", expand=True)
        .astype(float)
    )
    return df

df = split_params2(df)
df_global = split_params2(df_global)