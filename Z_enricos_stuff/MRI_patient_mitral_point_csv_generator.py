import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pyvista as pv

# ============================================================
# PARAMETRI
# ============================================================

ALL_PROCESSED_DIR = Path(
    "/home/rizzardi/Schreibtisch/AF001_aligned_processed"
)

OUTPUT_CSV = Path(
    "/home/rizzardi/Schreibtisch/subsampling_test/MRI_subsampling/patients_and_mitral_point2.csv"
)

# ============================================================
# ASSE APEX-BASE
# ============================================================

APEX_BASE_AXIS = np.array(
    [-1.0, 1.0, 0.0]
)

APEX_BASE_AXIS /= np.linalg.norm(
    APEX_BASE_AXIS
)

# ============================================================
# PESI
# ============================================================

W_PROJ  = 0.45
W_AREA  = 0.25
W_PLANE = 0.30

# ============================================================
# PATIENTS
# ============================================================

patient_dirs = sorted([
    p for p in ALL_PROCESSED_DIR.iterdir()
    if p.is_dir()
])

print(
    f"\nFound "
    f"{len(patient_dirs)} patients"
)

# ============================================================
# OUTPUT
# ============================================================

rows = []

# ============================================================
# LOOP
# ============================================================

for patient_dir in patient_dirs:

    patient_name = patient_dir.name

    print("\n====================================")
    print(patient_name)
    print("====================================")

    try:

        # ----------------------------------------------------
        # LOAD LV
        # ----------------------------------------------------

        lv_path = (
            patient_dir /
            "lv_endo-processed.vtp"
        )

        if not lv_path.exists():

            print("Missing LV")

            continue

        lv = pv.read(lv_path)

        # ----------------------------------------------------
        # CHECK PATCHES
        # ----------------------------------------------------

        if "isholepatch" not in lv.cell_data:

            print("No isholepatch")

            continue

        patch_mask = (
            lv.cell_data["isholepatch"] == 1
        )

        patches = lv.extract_cells(
            patch_mask
        )

        if patches.n_cells == 0:

            print("No patches")

            continue

        # ----------------------------------------------------
        # CONNECTIVITY
        # ----------------------------------------------------

        patches_conn = (
            patches.connectivity()
        )

        region_ids = np.unique(
            patches_conn.cell_data[
                "RegionId"
            ]
        )

        # ----------------------------------------------------
        # ANALYZE PATCHES
        # ----------------------------------------------------

        patch_infos = []

        for rid in region_ids:

            patch = patches_conn.threshold(
                [rid - 0.5, rid + 0.5],
                scalars="RegionId"
            )

            # ================================================
            # REPRESENTATIVE POINT
            # ================================================

            point = np.mean(
                patch.points,
                axis=0
            )

            x, y, z = point

            # ================================================
            # SCORES
            # ================================================

            projection = np.dot(
                point,
                APEX_BASE_AXIS
            )

            area = patch.area

            plane_value = x + y

            projection_score = np.clip(
                (projection + 1.0) / 2.0,
                0.0,
                1.0
            )

            area_score = np.clip(
                area / 0.20,
                0.0,
                1.0
            )

            plane_score = np.clip(
                (plane_value + 1.0) / 2.0,
                0.0,
                1.0
            )

            likelihood = (
                W_PROJ  * projection_score +
                W_AREA  * area_score +
                W_PLANE * plane_score
            )

            patch_infos.append({

                "rid": rid,

                "point": point,

                "likelihood": likelihood,
            })

        # ----------------------------------------------------
        # CHOOSE MITRAL
        # ----------------------------------------------------

        mitral_patch = max(
            patch_infos,
            key=lambda x: x["likelihood"]
        )

        point = mitral_patch["point"]

        # ----------------------------------------------------
        # SAVE ROW
        # ----------------------------------------------------

        rows.append({

            "patient": patient_name,

            "x": point[0],

            "y": point[1],

            "z": point[2],
        })

        print(
            f"Mitral point: "
            f"{point}"
        )

    except Exception as e:

        print(
            f"ERROR: {e}"
        )

# ============================================================
# SAVE CSV
# ============================================================

df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\n====================================")
print("CSV SAVED")
print(OUTPUT_CSV)
print("====================================")