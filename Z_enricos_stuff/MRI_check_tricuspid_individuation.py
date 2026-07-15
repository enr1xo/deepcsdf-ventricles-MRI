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
    "/home/rizzardi/Schreibtisch/AF001_aligned_processed"
)

APEX_BASE_AXIS = np.array([-1.0, 1.0, 0.0])
APEX_BASE_AXIS /= np.linalg.norm(APEX_BASE_AXIS)

PATIENTS_PER_PAGE = 64
GRID_ROWS = 8
GRID_COLS = 8

# ============================================================
# PESI LIKELIHOOD TRICUSPIDE
# ============================================================

W_PROJ = 0.1
W_AREA = 0.4
W_PLANE = 0.5

# ============================================================
# LISTA PAZIENTI
# ============================================================

patient_dirs = sorted([
    p for p in ALL_PROCESSED_DIR.iterdir()
    if p.is_dir()
])

print(f"Found {len(patient_dirs)} patients")

# ============================================================
# FUNZIONE CENTROIDE AREA-WEIGHTED
# ============================================================

def area_weighted_centroid(mesh):
    centers = mesh.cell_centers().points

    try:
        mesh_with_area = mesh.compute_cell_sizes(
            length=False,
            area=True,
            volume=False
        )

        areas = mesh_with_area.cell_data["Area"]

        if np.sum(areas) <= 0:
            return mesh.points.mean(axis=0)

        return np.average(
            centers,
            axis=0,
            weights=areas
        )

    except Exception:
        return mesh.points.mean(axis=0)


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

            rv_path = (
                patient_dir /
                "rv_endo-processed.vtp"
            )

            if not rv_path.exists():
                print(f"Missing RV: {patient_name}")
                continue

            rv = pv.read(rv_path)

            if "isholepatch" not in rv.cell_data:
                print(f"Missing isholepatch: {patient_name}")
                continue

            # ================================================
            # PATCHES RV
            # ================================================

            patches = rv.extract_cells(
                rv.cell_data["isholepatch"] == 1
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

                centroid = area_weighted_centroid(patch)

                x, y, z = centroid

                projection = np.dot(
                    centroid,
                    APEX_BASE_AXIS
                )

                area = patch.area

                # --------------------------------------------
                # PRIOR GEOMETRICO
                # --------------------------------------------
                # Come prima: x + y.
                # Se seleziona la polmonare invece della tricuspide,
                # prova x - y, -x, oppure x.
                plane_value = -z

                patch_infos.append({
                    "rid": rid,
                    "projection": projection,
                    "area": area,
                    "plane_value": plane_value,
                    "centroid": centroid,
                })

            # ================================================
            # NORMALIZZAZIONE SCORE
            # ================================================

            projections = np.array([
                p["projection"] for p in patch_infos
            ])

            areas = np.array([
                p["area"] for p in patch_infos
            ])

            plane_values = np.array([
                p["plane_value"] for p in patch_infos
            ])

            def normalize(values):
                vmin = values.min()
                vmax = values.max()

                if np.isclose(vmax, vmin):
                    return np.ones_like(values)

                return (values - vmin) / (vmax - vmin)

            projection_scores = normalize(projections)
            area_scores = normalize(areas)
            plane_scores = normalize(plane_values)

            # ================================================
            # LIKELIHOOD
            # ================================================

            for i, p in enumerate(patch_infos):

                likelihood = (
                    W_PROJ * projection_scores[i] +
                    W_AREA * area_scores[i] +
                    W_PLANE * plane_scores[i]
                )

                p["projection_score"] = projection_scores[i]
                p["area_score"] = area_scores[i]
                p["plane_score"] = plane_scores[i]
                p["likelihood"] = likelihood

            # ================================================
            # CHOOSE TRICUSPID
            # ================================================

            tricuspid_patch = max(
                patch_infos,
                key=lambda x: x["likelihood"]
            )

            tricuspid_region = tricuspid_patch["rid"]

            # ================================================
            # CREA LABEL
            # ================================================

            labels = np.zeros(
                rv.n_cells,
                dtype=np.int8
            )

            patch_region_ids = (
                patches_conn.cell_data["RegionId"]
            )

            patch_cell_ids = np.where(
                rv.cell_data["isholepatch"] == 1
            )[0]

            tricuspid_mask = (
                patch_region_ids == tricuspid_region
            )

            labels[
                patch_cell_ids[tricuspid_mask]
            ] = 1

            rv.cell_data["tricuspid_patch"] = labels

            # ================================================
            # ESTRAI TRICUSPIDE
            # ================================================

            tricuspid_cells = rv.extract_cells(
                rv.cell_data["tricuspid_patch"] == 1
            )

            # ================================================
            # PLOT
            # ================================================

            plotter.add_mesh(
                rv,
                color="lightgray",
                opacity=0.25,
            )

            plotter.add_mesh(
                tricuspid_cells,
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

    plotter.link_views()
    plotter.show()