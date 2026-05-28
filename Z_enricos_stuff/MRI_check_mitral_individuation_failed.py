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

PATIENTS_PER_PAGE = 16
GRID_ROWS = 4
GRID_COLS = 4

# asse apex -> base
APEX_BASE_AXIS = np.array([-1.0, 1.0, 0.0])
APEX_BASE_AXIS /= np.linalg.norm(APEX_BASE_AXIS)

# ============================================================
# PESI LIKELIHOOD
# ============================================================

W_PROJ = 0.45
W_AREA = 0.25
W_YX   = 0.3

# ============================================================
# LISTA PAZIENTI FOUND_AORTIC
# ============================================================

patients = [
    "AF045",
    "AF018",
    "LEU_BBB_21056",
    "LEU_BBB_21099",
    "LEU_BBB_21130",
    "LEU_BBB_21172",
    "LEU_BBB_21282",
    "LEU_BBB_21306",
    "LEU_BBB_21317",
    "LEU_BBB_21323",
    "LEU_BBB_21350",
    "LEU_BBB_21422",
    "LEU_NORM_0259",
    "LEU_NORM_0661",
    "LEU_NORM_0772",
    "LEU_NORM_0912",
    "LEU_NORM_1048",
    "LEU_NORM_1145",
    "LEU_NORM_1686",
    "LEU_NORM_2931",
    "LEU_NORM_3222",
    "LEU_NORM_3507",
    "LEU_NORM_4774",
    "LEU_NORM_4973",
    "LEU_NORM_5113",
    "LEU_NORM_5215",
    "LEU_NORM_5216",
    "LEU_NORM_F040",
    "LEU_NORM_F047",
    "LEU_NORM_F048",
    "S64",
    "VT002_MUG2",
    "VT022_MUG25",
    "yrm0832_v1",
    "yrm5856_v1",
]


# ============================================================
# PAGINAZIONE
# ============================================================

for page_start in range(
    0,
    len(patients),
    PATIENTS_PER_PAGE
):

    page_patients = patients[
        page_start :
        page_start + PATIENTS_PER_PAGE
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
        window_size=(2400, 2400),
    )

    # ========================================================
    # LOOP PAZIENTI
    # ========================================================

    for idx, patient in enumerate(page_patients):

        row = idx // GRID_COLS
        col = idx % GRID_COLS

        plotter.subplot(row, col)

        try:

            # =================================================
            # LOAD LV
            # =================================================

            patient_dir = (
                ALL_PROCESSED_DIR / patient
            )

            lv_candidates = list(
                patient_dir.rglob(
                    "lv_endo-processed.vtp"
                )
            )

            if len(lv_candidates) == 0:

                print(
                    f"Missing LV for {patient}"
                )

                continue

            lv_path = lv_candidates[0]

            lv = pv.read(lv_path)

            # =================================================
            # PATCHES
            # =================================================

            patches = lv.extract_cells(
                lv.cell_data["isholepatch"] == 1
            )

            if patches.n_cells == 0:

                print(
                    f"No patches for {patient}"
                )

                continue

            patches_conn = patches.connectivity()

            region_ids = np.unique(
                patches_conn.cell_data["RegionId"]
            )

            # =================================================
            # ANALISI PATCH
            # =================================================

            patch_infos = []

            for rid in region_ids:

                patch = patches_conn.threshold(
                    [rid - 0.5, rid + 0.5],
                    scalars="RegionId"
                )

                point = patch.points[0]

                x, y, z = point

                projection = np.dot(
                    point,
                    APEX_BASE_AXIS
                )

                area = patch.area

                yx_value = y + x

                patch_infos.append({
                    "rid": rid,
                    "projection": projection,
                    "area": area,
                    "yx_value": yx_value,
                    "point": point,
                })

            # =================================================
            # NORMALIZATION
            # =================================================

            areas = [
                p["area"]
                for p in patch_infos
            ]

            projections = [
                p["projection"]
                for p in patch_infos
            ]

            yx_values = [
                p["yx_value"]
                for p in patch_infos
            ]

            # areas_n = normalize(areas)

            # projections_n = normalize(
            #     projections
            # )

            # yx_n = normalize(yx_values)

            # =================================================
            # LIKELIHOOD
            # =================================================

            for p in patch_infos:

                projection = p["projection"]

                area = p["area"]

                plane_value = p["yx_value"]

                # ---------------------------------------------
                # PROJECTION SCORE
                # ---------------------------------------------

                # projection circa in [-1,1]
                projection_score = np.clip(
                    (projection + 1.0) / 2.0,
                    0.0,
                    1.0
                )

                # ---------------------------------------------
                # AREA SCORE
                # ---------------------------------------------

                # patch mitrale tipicamente ~0.15-0.20
                area_score = np.clip(
                    area / 0.20,
                    0.0,
                    1.0
                )

                # ---------------------------------------------
                # HALF-PLANE SCORE
                # ---------------------------------------------

                # x+y > 0 => sopra y=-x
                plane_score = np.clip(
                    (plane_value + 1.0) / 2.0,
                    0.0,
                    1.0
                )

                # ---------------------------------------------
                # FINAL LIKELIHOOD
                # ---------------------------------------------

                likelihood = (
                    W_PROJ * projection_score +
                    W_AREA * area_score +
                    W_YX   * plane_score
                )

                p["projection_score"] = projection_score
                p["area_score"] = area_score
                p["plane_score"] = plane_score

                p["likelihood"] = likelihood

                print(
                    f"rid={p['rid']} | "
                    f"proj={projection_score:.3f} | "
                    f"area={area_score:.3f} | "
                    f"plane={plane_score:.3f} | "
                    f"LIK={likelihood:.3f}"
                )
            # =================================================
            # MITRAL PATCH
            # =================================================

            mitral_patch = max(
                patch_infos,
                key=lambda x: x["likelihood"]
            )

            mitral_region = (
                mitral_patch["rid"]
            )

            # =================================================
            # CREA LABEL
            # =================================================

            labels = np.zeros(
                lv.n_cells,
                dtype=np.int8
            )

            patch_region_ids = (
                patches_conn.cell_data[
                    "RegionId"
                ]
            )

            patch_cell_ids = np.where(
                lv.cell_data[
                    "isholepatch"
                ] == 1
            )[0]

            mitral_mask = (
                patch_region_ids ==
                mitral_region
            )

            labels[
                patch_cell_ids[
                    mitral_mask
                ]
            ] = 1

            lv.cell_data[
                "mitral_patch"
            ] = labels

            # =================================================
            # ESTRAI MITRALE
            # =================================================

            mitral_cells = lv.extract_cells(
                lv.cell_data[
                    "mitral_patch"
                ] == 1
            )

            # =================================================
            # PLOT
            # =================================================

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
                patient,
                font_size=10,
            )

            # camera coerente
            plotter.view_vector(
                APEX_BASE_AXIS
            )

        except Exception as e:

            print(
                f"Error with {patient}: {e}"
            )

    # ========================================================
    # LINK CAMERAS
    # ========================================================

    plotter.link_views()

    # ========================================================
    # SHOW
    # ========================================================

    plotter.show()