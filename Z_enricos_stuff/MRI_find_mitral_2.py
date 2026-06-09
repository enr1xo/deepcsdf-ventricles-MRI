import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pyvista as pv

# ============================================================
# PARAMETRI
# ============================================================

patient = "LEU_NORM_4774"
patient= "AF052"
# patient = "LEU_NORM_F079"

ALL_PROCESSED_DIR = Path(
    "/home/rizzardi/Schreibtisch/AF001_aligned_processed"
)

ALL_PROCESSED_DIR = Path(r"C:\Users\e.rizzardi\OneDrive\Desktop")
patient = "AF001"

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
# LOAD LV
# ============================================================

patient_dir = ALL_PROCESSED_DIR / patient

lv_path = patient_dir / "lv_endo-processed.vtp"

lv = pv.read(lv_path)

print("\nLV loaded")
print(lv)

# ============================================================
# ESTRAI PATCH
# ============================================================

patches = lv.extract_cells(
    lv.cell_data["isholepatch"] == 1
)

print("\nPatch cells:", patches.n_cells)

# ============================================================
# CONNECTIVITY
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

    # punto rappresentativo
    point = patch.points[0]

    x, y, z = point

    # --------------------------------------------------------
    # PROIEZIONE APEX-BASE
    # --------------------------------------------------------

    projection = np.dot(
        point,
        APEX_BASE_AXIS
    )

    # --------------------------------------------------------
    # AREA PATCH
    # --------------------------------------------------------

    area = patch.area

    # --------------------------------------------------------
    # PRIOR GEOMETRICO
    # --------------------------------------------------------

    yx_value = y + x

    patch_infos.append({
        "rid": rid,
        "projection": projection,
        "area": area,
        "yx_value": yx_value,
        "point": point,
        "mesh": patch,
    })

    print(
        f"Region {rid} | "
        f"proj={projection:.4f} | "
        f"area={area:.4f} | "
        f"y-x={yx_value:.4f}"
    )

# ============================================================
# LIKELIHOOD
# ============================================================

print("\nPatch likelihoods:\n")

for p in patch_infos:

    projection = p["projection"]

    area = p["area"]

    plane_score = p["yx_value"]

    # --------------------------------------------------------
    # PROJECTION SCORE
    # --------------------------------------------------------

    projection_score = np.clip(
        (projection + 1.0) / 2.0,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # AREA SCORE
    # --------------------------------------------------------

    # area tipica mitrale ~0.15-0.20
    area_score = np.clip(
        area / 0.20,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Y-X SCORE
    # --------------------------------------------------------

    plane_score = np.clip(
        (plane_score + 1.0) / 2.0,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # FINAL LIKELIHOOD
    # --------------------------------------------------------

    likelihood = (
        W_PROJ * projection_score +
        W_AREA * area_score +
        W_YX   * plane_score
    )

    p["projection_score"] = projection_score
    p["area_score"] = area_score
    p["yx_score"] = plane_score

    p["likelihood"] = likelihood

    print(
        f"Region {p['rid']} | "
        f"proj={projection_score:.3f} | "
        f"area={area_score:.3f} | "
        f"yx={plane_score:.3f} | "
        f"LIK={likelihood:.3f}"
    )

# ============================================================
# SORT LIKELIHOODS
# ============================================================

patch_infos = sorted(
    patch_infos,
    key=lambda x: x["likelihood"],
    reverse=True
)

print("\nLIKELIHOOD RANKING\n")

for p in patch_infos:

    print(
        f"rid={p['rid']} | "
        f"LIK={p['likelihood']:.3f}"
    )

# ============================================================
# CHOOSE MITRAL PATCH
# ============================================================

mitral_patch_info = max(
    patch_infos,
    key=lambda x: x["likelihood"]
)

mitral_region = mitral_patch_info["rid"]

print(
    f"\nChosen mitral region: "
    f"{mitral_region}"
)

# ============================================================
# CREA LABEL
# ============================================================

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

# ============================================================
# ESTRAI PATCH MITRALE
# ============================================================

mitral_cells = lv.extract_cells(
    lv.cell_data["mitral_patch"] == 1
)

mitral_centroid = mitral_cells.points.mean(axis=0)

print("Mitral centroid:", mitral_centroid)

other_patch_mask = (
    (lv.cell_data["isholepatch"] == 1) &
    (lv.cell_data["mitral_patch"] == 0)
)
other_cells = lv.extract_cells(
    other_patch_mask
)
# ============================================================
# PLOT
# ============================================================

plotter = pv.Plotter()

# LV completa
plotter.add_mesh(
    lv,
    color="lightgray",
    opacity=0.25,
)

# patch mitrale
plotter.add_mesh(
    mitral_cells,
    color="red",
    show_edges=True,
    line_width=2,
)

# altre patch
plotter.add_mesh(
    other_cells,
    color="lightblue",
    opacity=0.9,
    show_edges=True,
)

centroid_sphere = pv.Sphere(
    radius=0.03,
    center=mitral_centroid
)

plotter.add_mesh(
    centroid_sphere,
    color="magenta",
    label="Mitral centroid"
)
# ============================================================
# PUNTI RAPPRESENTATIVI
# ============================================================

for p in patch_infos:

    sphere = pv.Sphere(
        radius=0.02,
        center=p["point"]
    )

    if p["rid"] == mitral_region:
        color = "yellow"
    else:
        color = "red"

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
    scale=0.3
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