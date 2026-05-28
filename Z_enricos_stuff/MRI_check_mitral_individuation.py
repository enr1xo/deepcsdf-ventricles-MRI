import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pyvista as pv

# ============================================================
# PARAMETRI
# ============================================================

ALL_PROCESSED_DIR = Path(
    "/home/rizzardi/Schreibtisch/all_processed_files"
)

ALL_PROCESSED_DIR = Path(
    "/home/rizzardi/Schreibtisch/reprocessed_noPatch"
)

APEX_BASE_AXIS = np.array([-1.0, 1.0, 0.0])
APEX_BASE_AXIS /= np.linalg.norm(APEX_BASE_AXIS)

PATIENTS_PER_PAGE = 64
GRID_ROWS = 8
GRID_COLS = 8

# ============================================================
# LISTA PAZIENTI
# ============================================================

patient_dirs = sorted([
    p for p in ALL_PROCESSED_DIR.iterdir()
    if p.is_dir()
])

print(f"Found {len(patient_dirs)} patients")

# ============================================================
# PAGINAZIONE
# ============================================================

for page_start in range(0, len(patient_dirs), PATIENTS_PER_PAGE):

    page_patients = patient_dirs[
        page_start : page_start + PATIENTS_PER_PAGE
    ]

    print(
        f"\nShowing patients "
        f"{page_start} -> "
        f"{page_start + len(page_patients)-1}"
    )

    # ========================================================
    # PLOTTER GRID
    # ========================================================

    plotter = pv.Plotter(
        shape=(GRID_ROWS, GRID_COLS),
        border=False,
        window_size=(3000, 3000),
    )

    # ========================================================
    # LOOP PAZIENTI
    # ========================================================

    for idx, patient_dir in enumerate(page_patients):

        row = idx // GRID_COLS
        col = idx % GRID_COLS

        plotter.subplot(row, col)

        patient_name = patient_dir.name

        try:

            lv_path = (
                patient_dir /
                "lv_endo-processed.vtp"
            )

            if not lv_path.exists():
                print(f"Missing LV: {patient_name}")
                continue

            lv = pv.read(lv_path)

            # ================================================
            # PATCHES
            # ================================================

            patches = lv.extract_cells(
                lv.cell_data["isholepatch"] == 1
            )

            if patches.n_cells == 0:
                print(f"No patches: {patient_name}")
                continue

            patches_conn = patches.connectivity()

            region_ids = np.unique(
                patches_conn.cell_data["RegionId"]
            )

            # ================================================
            # ANALISI PATCH
            # ================================================

            patch_infos = []

            for rid in region_ids:

                patch = patches_conn.threshold(
                    [rid - 0.5, rid + 0.5],
                    scalars="RegionId"
                )

                point = patch.points[0]

                projection = np.dot(
                    point,
                    APEX_BASE_AXIS
                )

                area = patch.area

                patch_infos.append({
                    "rid": rid,
                    "projection": projection,
                    "area": area,
                    "point": point,
                })

            # ================================================
            # LIKELIHOOD PARAMETERS
            # ================================================

            W_PROJ  = 0.45
            W_AREA  = 0.25
            W_PLANE = 0.3

            # ================================================
            # LIKELIHOOD
            # ================================================

            for p in patch_infos:

                point = p["point"]

                x, y, z = point

                projection = p["projection"]

                area = p["area"]

                # --------------------------------------------
                # HALF-PLANE PRIOR
                # --------------------------------------------

                # sopra y = -x
                plane_value = x + y

                # --------------------------------------------
                # PROJECTION SCORE
                # --------------------------------------------

                # projection circa in [-1,1]
                projection_score = np.clip(
                    (projection + 1.0) / 2.0,
                    0.0,
                    1.0
                )

                # --------------------------------------------
                # AREA SCORE
                # --------------------------------------------

                # patch mitrale tipicamente ~0.15-0.20
                area_score = np.clip(
                    area / 0.20,
                    0.0,
                    1.0
                )

                # --------------------------------------------
                # HALF-PLANE SCORE
                # --------------------------------------------

                # x+y > 0 => sopra y=-x
                plane_score = np.clip(
                    (plane_value + 1.0) / 2.0,
                    0.0,
                    1.0
                )

                # --------------------------------------------
                # FINAL LIKELIHOOD
                # --------------------------------------------

                likelihood = (
                    W_PROJ  * projection_score +
                    W_AREA  * area_score +
                    W_PLANE * plane_score
                )

                p["projection_score"] = projection_score
                p["area_score"] = area_score
                p["plane_score"] = plane_score

                p["likelihood"] = likelihood

            # ================================================
            # CHOOSE MITRAL
            # ================================================

            mitral_patch = max(
                patch_infos,
                key=lambda x: x["likelihood"]
            )

            mitral_region = mitral_patch["rid"]

            # ================================================
            # CREA LABEL
            # ================================================

            labels = np.zeros(
                lv.n_cells,
                dtype=np.int8
            )

            patch_region_ids = (
                patches_conn.cell_data["RegionId"]
            )

            patch_cell_ids = np.where(
                lv.cell_data["isholepatch"] == 1
            )[0]

            mitral_mask = (
                patch_region_ids == mitral_region
            )

            labels[
                patch_cell_ids[mitral_mask]
            ] = 1

            lv.cell_data["mitral_patch"] = labels

            # ================================================
            # ESTRAI MITRALE
            # ================================================

            mitral_cells = lv.extract_cells(
                lv.cell_data["mitral_patch"] == 1
            )

            # ================================================
            # PLOT
            # ================================================

            plotter.add_mesh(
                lv,
                color="lightgray",
                opacity=0.25,
            )

            plotter.add_mesh(
                mitral_cells,
                color="red",
                show_edges=False,
            )

            plotter.add_text(
                patient_name,
                font_size=8,
            )

            plotter.camera_position = "xy"

        except Exception as e:

            print(
                f"Error with {patient_name}: {e}"
            )

    # ========================================================
    # SHOW PAGE
    # ========================================================

    plotter.link_views()
    plotter.show()
    