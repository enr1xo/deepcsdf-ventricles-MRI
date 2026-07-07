"""
in this code we identify the apex as the point in the LV that is furthest from the mitral valve centroid.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pyvista as pv

# ============================================================
# PARAMETRI
# ============================================================

ALL_PROCESSED_DIR = Path(r"C:\Users\e.rizzardi\OneDrive\Desktop\processed_patients")
patient = "AF001"

# asse apex -> base iniziale / approssimato
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
        (plane_score + 1.0) / 2.0,
        0.0,
        1.0
    )

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

patch_region_ids = patches_conn.cell_data["RegionId"]

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


# ============================================================
# CONTROLLO OMOGENEITÀ AREE TRIANGOLI PATCH MITRALE
# ============================================================

mitral_surf = mitral_cells.extract_surface().triangulate()

cell_sizes = mitral_surf.compute_cell_sizes(
    length=False,
    area=True,
    volume=False
)

areas = cell_sizes.cell_data["Area"]

print("\nMitral patch triangle areas")
print("n triangles:", len(areas))
print("min area:", areas.min())
print("max area:", areas.max())
print("mean area:", areas.mean())
print("std area:", areas.std())
print("coefficient of variation:", areas.std() / areas.mean())
print("max / min:", areas.max() / areas.min())


mitral_centroid = mitral_cells.points.mean(axis=0)

print("\nMitral centroid:", mitral_centroid)


# surf = mitral_cells.extract_surface().triangulate()

# faces = surf.faces.reshape(-1, 4)[:, 1:]

# points = surf.points

# weighted_sum = np.zeros(3)
# total_area = 0.0

# for tri in faces:

#     p0, p1, p2 = points[tri]

#     area = (
#         np.linalg.norm(
#             np.cross(p1 - p0, p2 - p0)
#         ) / 2.0
#     )

#     tri_centroid = (p0 + p1 + p2) / 3.0

#     weighted_sum += area * tri_centroid
#     total_area += area

# area_weighted_centroid = (
#     weighted_sum / total_area
# )

# print("Point centroid :", mitral_centroid)
# print("Area centroid  :", area_weighted_centroid)

# print(
#     "Distance:",
#     np.linalg.norm(
#         area_weighted_centroid -
#         mitral_centroid
#     )
# )

# ============================================================
# TROVA APEX
# metodo: punto LV più distante dal centroide mitralico
# ============================================================

lv_points = lv.points

distances_from_mitral = np.linalg.norm(
    lv_points - mitral_centroid,
    axis=1
)

apex_idx = np.argmax(distances_from_mitral)
apex_point = lv_points[apex_idx]

print("Apex index:", apex_idx)
print("Apex point:", apex_point)
print("Distance apex-mitral:", distances_from_mitral[apex_idx])

# asse anatomico stimato: apex -> base/mitrale
estimated_apex_base_axis = mitral_centroid - apex_point
estimated_apex_base_axis /= np.linalg.norm(estimated_apex_base_axis)

print("Estimated apex-base axis:", estimated_apex_base_axis)

# ============================================================
# ALTRE PATCH
# ============================================================

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

plotter.add_mesh(
    lv,
    color="lightgray",
    opacity=0.25,
)

plotter.add_mesh(
    mitral_cells,
    color="red",
    show_edges=True,
    line_width=2,
)

plotter.add_mesh(
    other_cells,
    color="lightblue",
    opacity=0.9,
    show_edges=True,
)

# centroide mitrale
centroid_sphere = pv.Sphere(
    radius=0.02,
    center=mitral_centroid
)

plotter.add_mesh(
    centroid_sphere,
    color="magenta"
)

# apex
apex_sphere = pv.Sphere(
    radius=0.02,
    center=apex_point
)

plotter.add_mesh(
    apex_sphere,
    color="cyan"
)

# linea apex-centroide mitrale
apex_base_line = pv.Line(
    apex_point,
    mitral_centroid
)

plotter.add_mesh(
    apex_base_line,
    color="yellow",
    line_width=6
)

# ============================================================
# PUNTI RAPPRESENTATIVI PATCH
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

# origin = pv.Sphere(
#     radius=0.02,
#     center=(0, 0, 0)
# )

# plotter.add_mesh(
#     origin,
#     color="red"
# )

# ============================================================
# ASSE APEX-BASE STIMATO
# ============================================================

arrow = pv.Arrow(
    start=apex_point,
    direction=estimated_apex_base_axis,
    scale=0.4
)

plotter.add_mesh(
    arrow,
    color="green"
)

# ============================================================
# LABELS
# ============================================================

plotter.add_point_labels(
    [mitral_centroid],
    ["Mitral centroid"],
    font_size=18
)

plotter.add_point_labels(
    [apex_point],
    ["Apex"],
    font_size=18
)

# ============================================================
# ASSI XYZ + COORDINATE
# ============================================================

plotter.show_bounds(
    grid="front",
    location="outer",
    all_edges=True,
)

plotter.add_axes()
plotter.show()