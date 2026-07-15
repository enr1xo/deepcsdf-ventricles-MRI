"""
in questo codice leggiamo i due csv in cui abbiamo
    - le posizioni del centroide di massa della mitrale (C_area) e dell'apex (max_D)
    - la posizione del centroide di massa della tricuspide

scriviamo un nuovo csv in cui mettiamo le posizioni di tutti e 3.
Questo csv lo leggeremo poi per costruire i 3 assi normali ai piani di acquisizione.


occhio ai nomi con cui sono chiamati i punti!!!!!!!!
usiamo solo i pazienti di cui conosciamo già la mitrale e l'apice, nell'altro sono inclus anche quelli esclusi
"""

from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

OUTPUT_DIR = Path("/home/rizzardi/Schreibtisch/MRI_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

final_csv_name = "mitral_apex_tricuspid_locations.csv"
final_csv_path = OUTPUT_DIR / final_csv_name

mitral_and_apex_path = Path(
    "/home/rizzardi/Schreibtisch/MRI_model/mitral_Carea_apex_MaxD.csv"
)

tricuspid_path = Path(
    "/home/rizzardi/Schreibtisch/MRI_model/tricuspid_centroids_distance_study.csv"
)


# ============================================================
# COLUMN NAMES
# ============================================================

mitral_centroid_names = [
    "C_area_x",
    "C_area_y",
    "C_area_z",
]

apex_names = [
    "A_maxD_x",
    "A_maxD_y",
    "A_maxD_z",
]

tricuspid_centroid_names = [
    "mass_centroid_x",
    "mass_centroid_y",
    "mass_centroid_z",
]

tricuspid_final_names = [
    "T_area_x",
    "T_area_y",
    "T_area_z",
]


# ============================================================
# READ CSV
# ============================================================

mitral_apex_df = pd.read_csv(
    mitral_and_apex_path,
    sep=None,
    engine="python"
)

tricuspid_df = pd.read_csv(
    tricuspid_path,
    sep=None,
    engine="python"
)

mitral_apex_df["patient_id"] = mitral_apex_df["patient_id"].astype(str)
tricuspid_df["patient_id"] = tricuspid_df["patient_id"].astype(str)


# ============================================================
# CHECK COLUMNS
# ============================================================

required_mitral_apex_cols = (
    ["patient_id"]
    + mitral_centroid_names
    + apex_names
)

required_tricuspid_cols = (
    ["patient_id", "status"]
    + tricuspid_centroid_names
)

missing_mitral_apex = [
    c for c in required_mitral_apex_cols
    if c not in mitral_apex_df.columns
]

missing_tricuspid = [
    c for c in required_tricuspid_cols
    if c not in tricuspid_df.columns
]

if missing_mitral_apex:
    raise ValueError(
        f"Missing columns in mitral/apex CSV: {missing_mitral_apex}"
    )

if missing_tricuspid:
    raise ValueError(
        f"Missing columns in tricuspid CSV: {missing_tricuspid}"
    )


# ============================================================
# USA COME RIFERIMENTO SOLO I PAZIENTI MITRALE + APICE
# ============================================================

valid_patient_ids = set(
    mitral_apex_df["patient_id"]
)

tricuspid_ok_df = tricuspid_df[
    (tricuspid_df["status"] == "ok") &
    (tricuspid_df["patient_id"].isin(valid_patient_ids))
].copy()


# ============================================================
# SELECT USEFUL COLUMNS
# ============================================================

mitral_apex_small = mitral_apex_df[
    required_mitral_apex_cols
].copy()

tricuspid_small = tricuspid_ok_df[
    ["patient_id"] + tricuspid_centroid_names
].copy()

tricuspid_small = tricuspid_small.rename(
    columns={
        old: new
        for old, new in zip(
            tricuspid_centroid_names,
            tricuspid_final_names
        )
    }
)


# ============================================================
# MERGE
# ============================================================
# how="left" mantiene tutti e soli i pazienti del file mitrale/apice.
# Se manca la tricuspide valida per un paziente, T_area_* sarà NaN.

final_df = mitral_apex_small.merge(
    tricuspid_small,
    on="patient_id",
    how="left"
)


# ============================================================
# FINAL ORDER
# ============================================================

final_columns = (
    ["patient_id"]
    + mitral_centroid_names
    + apex_names
    + tricuspid_final_names
)

final_df = final_df[
    final_columns
]


# ============================================================
# SAVE
# ============================================================

final_df.to_csv(
    final_csv_path,
    index=False,
    sep=";"
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\nDone.")
print(f"Saved CSV to: {final_csv_path}")

print(f"\nRows mitral/apex reference: {len(mitral_apex_df)}")
print(f"Rows tricuspid total:       {len(tricuspid_df)}")
print(f"Rows tricuspid ok + valid:  {len(tricuspid_ok_df)}")
print(f"Rows final:                {len(final_df)}")

missing_tricuspid = final_df[
    tricuspid_final_names
].isna().any(axis=1).sum()

print(f"Patients without valid tricuspid: {missing_tricuspid}")

print("\nFinal columns:")
print(list(final_df.columns))

print("\nPreview:")
print(final_df.head())