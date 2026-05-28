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

# ============================================================
# SOLO PAZIENTI:
# - small_patch == 1
# - no_holes == 1
# ============================================================

patients = [

    # ------------------------
    # no_holes
    # ------------------------

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

    # ------------------------
    # small_patch
    # ------------------------

    "LEU_BBB_21248",
    "LEU_NORM_0016",
    "LEU_NORM_1709",
    "LEU_NORM_2232",
    "LEU_NORM_2994",
    "LEU_NORM_F080",
    "LEU_NORM_F124",
    "yrm8967_v1",
]

print(f"\nTesting {len(patients)} problematic patients")

# ============================================================
# AXIS
# ============================================================

APEX_BASE_AXIS = np.array([-1.0, 1.0, 0.0])
APEX_BASE_AXIS /= np.linalg.norm(APEX_BASE_AXIS)

# ============================================================
# WEIGHTS
# ============================================================

W_PROJ  = 0.45
W_AREA  = 0.25
W_PLANE = 0.3

# ============================================================
# PLOTTER
# ============================================================

plotter = pv.Plotter(
    shape=(6, 7),
    window_size=(3200, 2600),
)

# ============================================================
# LOOP
# ============================================================

for idx, patient in enumerate(patients):

    row = idx // 7
    col = idx % 7

    plotter.subplot(row, col)

    print("\n================================================")
    print(patient)
    print("================================================")

    try:

        # ----------------------------------------------------
        # LOAD LV
        # ----------------------------------------------------

        patient_dir = (
            ALL_PROCESSED_DIR / patient
        )

        lv_candidates = list(
            patient_dir.rglob(
                "lv_endo-processed.vtp"
            )
        )

        if len(lv_candidates) == 0:

            print("LV NOT FOUND")

            plotter.add_text(
                f"{patient}\nNO LV",
                font_size=10,
                color="red"
            )

            continue

        lv_path = lv_candidates[0]

        print(f"LV path: {lv_path}")

        lv = pv.read(lv_path)

        # ----------------------------------------------------
        # CHECK ISHOLEPATCH
        # ----------------------------------------------------

        if "isholepatch" not in lv.cell_data:

            print("NO ISHOLEPATCH ARRAY")

            plotter.add_mesh(
                lv,
                color="orange",
                opacity=0.5,
            )

            plotter.add_text(
                f"{patient}\nNO ARRAY",
                font_size=10,
                color="red"
            )

            continue

        # ----------------------------------------------------
        # PATCHES
        # ----------------------------------------------------

        patch_mask = (
            lv.cell_data["isholepatch"] == 1
        )

        n_patch_cells = np.sum(
            patch_mask
        )

        print(
            f"Patch cells: "
            f"{n_patch_cells}"
        )

        patches = lv.extract_cells(
            patch_mask
        )

        # ----------------------------------------------------
        # NO PATCH CASE
        # ----------------------------------------------------

        if patches.n_cells == 0:

            print("NO PATCH FOUND")

            boundary_edges = lv.extract_feature_edges(
                boundary_edges=True,
                feature_edges=False,
                manifold_edges=False,
                non_manifold_edges=False,
            )

            print(
                f"Boundary edges: "
                f"{boundary_edges.n_cells}"
            )

            plotter.add_mesh(
                lv,
                color="lightgray",
                opacity=0.3,
            )

            if boundary_edges.n_cells > 0:

                plotter.add_mesh(
                    boundary_edges,
                    color="red",
                    line_width=4,
                )

            plotter.add_text(
                f"{patient}\nNO PATCH",
                font_size=9,
                color="red"
            )

            continue

        # ----------------------------------------------------
        # CONNECTIVITY
        # ----------------------------------------------------

        patches_conn = patches.connectivity()

        region_ids = np.unique(
            patches_conn.cell_data[
                "RegionId"
            ]
        )

        print(
            f"Regions: {region_ids}"
        )

        # ----------------------------------------------------
        # PATCH ANALYSIS
        # ----------------------------------------------------

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

            plane_value = x + y

            # --------------------------------------------
            # SCORES
            # --------------------------------------------

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
                "likelihood": likelihood,
            })

            print(
                f"Region {rid} | "
                f"LIK={likelihood:.3f}"
            )

        # ----------------------------------------------------
        # CHOOSE MITRAL
        # ----------------------------------------------------

        mitral_patch = max(
            patch_infos,
            key=lambda x: x["likelihood"]
        )

        mitral_region = (
            mitral_patch["rid"]
        )

        print(
            f"Chosen region: "
            f"{mitral_region}"
        )

        # ----------------------------------------------------
        # LABELS
        # ----------------------------------------------------

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
            patch_mask
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

        # ----------------------------------------------------
        # EXTRACT MITRAL
        # ----------------------------------------------------

        mitral_cells = lv.extract_cells(
            lv.cell_data[
                "mitral_patch"
            ] == 1
        )

        # ----------------------------------------------------
        # PLOT
        # ----------------------------------------------------

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
            font_size=9,
        )

        plotter.camera_position = "xy"

    except Exception as e:

        print(f"ERROR: {e}")

        plotter.add_text(
            f"{patient}\nERROR",
            font_size=10,
            color="red"
        )

# ============================================================
# LINK CAMERAS
# ============================================================

plotter.link_views()

# ============================================================
# SHOW
# ============================================================

plotter.show()