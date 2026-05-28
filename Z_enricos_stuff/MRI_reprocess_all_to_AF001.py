import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pyvista as pv
from loguru import logger
from tqdm import tqdm

from ventricles_data_preparing_MRI import (
    extract_processed_ventricle_surfaces,
    resolve_patient_dir,
    ensure_vtu_exists,
)

# ============================================================
# PATHS
# ============================================================

SOURCE_DIR = Path(
    "/home/rizzardi/Schreibtisch/all_processed_files"
)

SAVE_DIR = Path(
    "/home/rizzardi/Schreibtisch/AF001_aligned_processed"
)

SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REFERENCE_PATIENT = "AF001"

# ============================================================
# LOAD REFERENCE MESH
# ============================================================

ref_dir = resolve_patient_dir(
    SOURCE_DIR,
    REFERENCE_PATIENT
)

ref_vtu = ensure_vtu_exists(
    ref_dir
)

reference_mesh = pv.read(
    ref_vtu
)

print()
print("Reference patient:", REFERENCE_PATIENT)
print("Reference mesh:", ref_vtu)

# ============================================================
# PATIENT LIST
# ============================================================

patients = sorted([
    p.name
    for p in SOURCE_DIR.iterdir()
    if p.is_dir()
    and "single_patients_100000pts" not in p.name
])

print()
print(f"Found {len(patients)} patients")

# ============================================================
# REPROCESS
# ============================================================

failed = []

for patient in tqdm(patients):

    print("\n====================================")
    print(patient)
    print("====================================")

    try:

        extracted = extract_processed_ventricle_surfaces(
            patient_name=patient,
            reference_name=REFERENCE_PATIENT,
            reference_mesh=reference_mesh,
            source_dir=SOURCE_DIR,
        )

        epicardium = extracted["epicardium_surface"]
        lv = extracted["LV_endo_surface"]
        rv = extracted["RV_endo_surface"]
        original = extracted["original_mesh"]

        patient_save_dir = SAVE_DIR / patient

        patient_save_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        epicardium.save(
            patient_save_dir /
            "epicardium-processed.vtp"
        )

        lv.save(
            patient_save_dir /
            "lv_endo-processed.vtp"
        )

        rv.save(
            patient_save_dir /
            "rv_endo-processed.vtp"
        )

        original.save(
            patient_save_dir /
            f"{patient}.vtu"
        )

        if "isholepatch" in lv.cell_data:
            n_lv_patch = int(
                (lv.cell_data["isholepatch"] == 1).sum()
            )
        else:
            n_lv_patch = -1

        print(
            f"Saved {patient} | "
            f"LV patch cells = {n_lv_patch}"
        )

    except Exception as e:

        logger.error(
            f"Failed {patient}: {e}"
        )

        failed.append(patient)

# ============================================================
# REPORT
# ============================================================

print("\n====================================")
print("DONE")
print("====================================")

print("Failed patients:")

for p in failed:
    print(p)

print(f"\nTotal failed: {len(failed)}")