import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from pathlib import Path
import pyvista as pv
from loguru import logger
import numpy as np

from ventricles_data_preparing_MRI import (
    extract_processed_ventricle_surfaces,
    resolve_patient_dir,
)

from config import PATIENT_MESHES_DIR

# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = Path(
    "/home/rizzardi/Schreibtisch/reprocessed_noPatch"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# PATIENTS
# ============================================================

patients = [

    "LEU_NORM_0093",
    "LEU_NORM_0418",
    "LEU_NORM_1236",
    "LEU_NORM_1424",
    "LEU_NORM_2176",
    "LEU_NORM_2347",
    "LEU_NORM_2614",
    "LEU_NORM_2984",
    "LEU_NORM_3301",
    "LEU_NORM_3302",
    "LEU_NORM_3407",
    "LEU_NORM_3825",
    "LEU_NORM_4421",
    "LEU_NORM_4695",
    "LEU_NORM_5028",
    "LEU_NORM_5077",
    "LEU_NORM_F009",
    "LEU_NORM_F015",
    "LEU_NORM_F019",
    "LEU_NORM_F022",
    "LEU_NORM_F029",
    "LEU_NORM_F057",
    "LEU_NORM_F071",
    "LEU_NORM_F083",
    "LEU_NORM_F107",
    "S67",
    "VT0010_MUG7",
    "VT014_MUG12",
    "VT017_MUG15",
    "VT019_MUG17",
    "yrm3751_v1",
    "yrm4015_v1",
    "yrm7026_v1_1",
]

# ============================================================
# REFERENCE
# ============================================================

reference_patient = "AF001"

ref_dir = resolve_patient_dir(
    PATIENT_MESHES_DIR,
    reference_patient
)

ref_vtu = next(
    ref_dir.rglob("*.vtu")
)

reference_mesh = pv.read(
    ref_vtu
)

# ============================================================
# LOOP
# ============================================================

for patient in patients:

    print("\n====================================")
    print(patient)
    print("====================================")

    try:

        extracted = (
            extract_processed_ventricle_surfaces(
                patient_name=patient,
                reference_name=reference_patient,
                reference_mesh=reference_mesh,
                source_dir=PATIENT_MESHES_DIR,
            )
        )

        epi = extracted["epicardium_surface"]
        lv  = extracted["LV_endo_surface"]
        rv  = extracted["RV_endo_surface"]

        # ----------------------------------------------------
        # SAVE DIR
        # ----------------------------------------------------

        save_dir = OUTPUT_DIR / patient

        save_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------
        print(
            patient,
            np.unique(
                lv.cell_data["isholepatch"]
            )
        )

        print(
            "patch cells:",
            np.sum(
                lv.cell_data["isholepatch"] == 1
            )
        )
        
        epi.save(
            save_dir /
            "epicardium-processed.vtp"
        )

        lv.save(
            save_dir /
            "lv_endo-processed.vtp"
        )

        rv.save(
            save_dir /
            "rv_endo-processed.vtp"
        )

        print("saved")

        # ----------------------------------------------------
        # DEBUG PATCHES
        # ----------------------------------------------------

        if "isholepatch" in lv.cell_data:

            n_patch = np.sum(
                lv.cell_data["isholepatch"] == 1
            )

            print(
                f"LV patch cells: "
                f"{n_patch}"
            )

    except Exception as e:

        logger.error(
            f"{patient} failed: {e}"
        )