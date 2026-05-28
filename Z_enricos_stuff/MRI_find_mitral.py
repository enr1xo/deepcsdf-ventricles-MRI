import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pyvista as pv

from config import PATIENT_MESHES_DIR

from ventricles_data_preparing import (
    ensure_vtu_exists,
    extract_processed_ventricle_surfaces,
)

# ============================================================
# PARAMETRI
# ============================================================

reference_patient = "AF001"

patient= reference_patient
# patient = "LEU_NORM_0093"
# patient = "LEU_BBB_21070"
# patient = "LEU_NORM_3864"
# patient = "S68"
# patient = "VT001_MUG1"
# patient = "yrm0348_v2"

# asse apex -> base noto dopo alignment
APEX_BASE_AXIS = np.array([-1.0, 1.0, 0.0])
APEX_BASE_AXIS /= np.linalg.norm(APEX_BASE_AXIS)

# ============================================================
# LOAD REFERENCE
# ============================================================
all_processed_files = "/home/rizzardi/Schreibtisch/all_processed_files"

reference_root = Path(all_processed_files) / reference_patient
reference_vtu = ensure_vtu_exists(reference_root)

reference_mesh = pv.read(reference_vtu)

# ============================================================
# ESTRAZIONE SUPERFICI PROCESSATE
# ============================================================

patient_dir = Path(PATIENT_MESHES_DIR) / patient

lv_path = patient_dir / "lv_endo-processed.vtp"

lv = pv.read(lv_path)

print("\nLV loaded")
print(lv)

# ============================================================
# ESTRAI SOLO PATCH AGGIUNTI
# ============================================================

patches = lv.extract_cells(
    lv.cell_data["isholepatch"] == 1
)

print("\nPatch cells:", patches.n_cells)

# ============================================================
# CONNECTIVITY SUI PATCH
# ============================================================

patches_conn = patches.connectivity()

region_ids = np.unique(
    patches_conn.cell_data["RegionId"]
)

print("\nFound regions:", region_ids)

# ============================================================
# ANALISI PATCH
# ============================================================

patch_infos = []

for rid in region_ids:

    patch = patches_conn.threshold(
        [rid - 0.5, rid + 0.5],
        scalars="RegionId"
    )

    # prende un punto qualsiasi
    representative_point = patch.points[0]

    projection = np.dot(
        representative_point,
        APEX_BASE_AXIS
    )

    area = patch.area

    patch_infos.append({
        "rid": rid,
        "projection": projection,
        "area": area,
        "point": representative_point,
        "mesh": patch
    })

    print(
        f"Region {rid} | "
        f"proj = {projection:.4f} | "
        f"area = {area:.4f}"
    )

# ============================================================
# ORDINA PER POSIZIONE BASALE
# ============================================================

patch_infos = sorted(
    patch_infos,
    key=lambda x: x["projection"],
    reverse=True
)

# prendiamo i 2 più basali
top_candidates = patch_infos[:2]

print("\nTop basal candidates:")
for p in top_candidates:
    print(
        f"rid={p['rid']} "
        f"proj={p['projection']:.4f} "
        f"area={p['area']:.4f}"
    )

# ============================================================
# MITRALE = area maggiore tra i più basali
# ============================================================

mitral_patch_info = max(
    top_candidates,
    key=lambda x: x["area"]
)

mitral_region = mitral_patch_info["rid"]

print("\nChosen mitral region:", mitral_region)

# ============================================================
# CREA LABEL
# ============================================================

labels = np.zeros(lv.n_cells, dtype=np.int8)

patch_region_ids = patches_conn.cell_data["RegionId"]

# ATTENZIONE:
# patches_conn ha solo le celle patch,
# quindi dobbiamo ricostruire gli indici originali

patch_cell_ids = np.where(
    lv.cell_data["isholepatch"] == 1
)[0]

mitral_patch_cells = (
    patch_region_ids == mitral_region
)

labels[
    patch_cell_ids[mitral_patch_cells]
] = 1

lv.cell_data["mitral_patch"] = labels

# ============================================================
# PLOT
# ============================================================

plotter = pv.Plotter()

# mesh completa
plotter.add_mesh(
    lv,
    color="lightgray",
    opacity=0.25,
)

# patch mitrale
mitral_cells = lv.extract_cells(
    lv.cell_data["mitral_patch"] == 1
)

plotter.add_mesh(
    mitral_cells,
    color="red",
    show_edges=True,
    line_width=2,
)

# punti rappresentativi
for p in patch_infos:

    sphere = pv.Sphere(
        radius=0.01,
        center=p["point"]
    )

    if p["rid"] == mitral_region:
        color = "yellow"
    else:
        color = "blue"

    plotter.add_mesh(
        sphere,
        color=color
    )

# ============================================================
# ORIGINE
# ============================================================

origin = pv.Sphere(
    radius=0.02,
    center=(0, 0, 0)
)

plotter.add_mesh(
    origin,
    color="red"
)

# ============================================================
# ASSE APEX-BASE
# ============================================================

axis_length = 2.0

p0 = -axis_length * APEX_BASE_AXIS
p1 =  axis_length * APEX_BASE_AXIS

axis_line = pv.Line(p0, p1)

plotter.add_mesh(
    axis_line,
    color="green",
    line_width=5,
)

# freccia
arrow = pv.Arrow(
    start=(0, 0, 0),
    direction=APEX_BASE_AXIS,
    scale=0.5
)

plotter.add_mesh(
    arrow,
    color="yellow"
)

# ============================================================
# ASSI XYZ + COORDINATE
# ============================================================

plotter.show_bounds(
    grid='front',
    location='outer',
    all_edges=True,
)

plotter.add_axes()

plotter.show()