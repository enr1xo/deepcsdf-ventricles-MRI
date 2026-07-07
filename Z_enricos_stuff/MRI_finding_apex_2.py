"""
In this code, we identify the mitral patch and compute its centroid.
Then we estimate the LV apex using two methods:

1. Max-distance method:
   the apex is the LV point farthest from the mitral centroid.

2. PCA method:
   the LV long axis is estimated as the first principal component of the LV points.
   The axis is oriented from the mitral centroid toward the apex, and the apex is
   selected as the LV point with the largest projection along this direction.

Both apex estimates are plotted for visual comparison.
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
patient = "LEU_NORM_0016"
# patietn = "LEU_BBB_21001"
# patient = "S62"
# patient = "VT0010_MUG7"
# patient = "yrm0342_v1"
patient = "LEU_BBB_21065"

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

# mitral_surf = mitral_cells.extract_surface().triangulate()

# cell_sizes = mitral_surf.compute_cell_sizes(
#     length=False,
#     area=True,
#     volume=False
# )

# areas = cell_sizes.cell_data["Area"]

# print("\nMitral patch triangle areas")
# print("n triangles:", len(areas))
# print("min area:", areas.min())
# print("max area:", areas.max())
# print("mean area:", areas.mean())
# print("std area:", areas.std())
# print("coefficient of variation:", areas.std() / areas.mean())
# print("max / min:", areas.max() / areas.min())

mitral_centroid_points = mitral_cells.points.mean(axis=0)

# print("\nMitral centroid:", mitral_centroid)


surf = mitral_cells.extract_surface().triangulate()

faces = surf.faces.reshape(-1, 4)[:, 1:]

points = surf.points

weighted_sum = np.zeros(3)
total_area = 0.0

for tri in faces:

    p0, p1, p2 = points[tri]

    area = (
        np.linalg.norm(
            np.cross(p1 - p0, p2 - p0)
        ) / 2.0
    )

    tri_centroid = (p0 + p1 + p2) / 3.0

    weighted_sum += area * tri_centroid
    total_area += area

area_weighted_centroid = (
    weighted_sum / total_area
)

mitral_centroid_area = area_weighted_centroid

mitral_centroid = mitral_centroid_points

print("Point centroid :", mitral_centroid)
print("Area centroid  :", area_weighted_centroid)

print(
    "Distance:",
    np.linalg.norm(
        area_weighted_centroid -
        mitral_centroid
    )
)

# ============================================================
# TROVA APEX
# metodo 1: punto LV più distante dal centroide mitralico
# ============================================================

lv_points = lv.points

distances_from_mitral = np.linalg.norm(
    lv_points - mitral_centroid,
    axis=1
)

apex_dist_idx = np.argmax(distances_from_mitral)
apex_dist_point = lv_points[apex_dist_idx]

print("\nApex by max distance")
print("Apex index:", apex_dist_idx)
print("Apex point:", apex_dist_point)
print("Distance apex-mitral:", distances_from_mitral[apex_dist_idx])

axis_dist = mitral_centroid - apex_dist_point
axis_dist /= np.linalg.norm(axis_dist)

print("Axis by max distance:", axis_dist)

# ============================================================
# TROVA APEX
# metodo 2: PCA + proiezione lungo il primo autovettore
# ============================================================

center_lv = lv_points.mean(axis=0)
X = lv_points - center_lv

cov = np.cov(X.T)

eigvals, eigvecs = np.linalg.eigh(cov)

print("\nPCA eigenvalues:", eigvals)

idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

long_axis = eigvecs[:, 0]

# orienta l'asse dalla mitrale verso l'apice
# se punta verso il centro LV, lo invertiamo
v_base_to_center = center_lv - mitral_centroid

if np.dot(long_axis, v_base_to_center) < 0:
    long_axis *= -1

projections = np.dot(
    lv_points - mitral_centroid,
    long_axis
)

apex_pca_idx = np.argmax(projections)
apex_pca_point = lv_points[apex_pca_idx]

print("\nApex by PCA")
print("Eigenvalues:", eigvals)
print("Long axis:", long_axis)
print("Apex index:", apex_pca_idx)
print("Apex point:", apex_pca_point)
print("Projection:", projections[apex_pca_idx])

axis_pca = mitral_centroid - apex_pca_point
axis_pca /= np.linalg.norm(axis_pca)

print("Axis by PCA:", axis_pca)

print(
    "\nDistance between apex methods:",
    np.linalg.norm(apex_dist_point - apex_pca_point)
)
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
centroid_points_sphere = pv.Sphere(
    radius=0.015,
    center=mitral_centroid_points
)

centroid_area_sphere = pv.Sphere(
    radius=0.015,
    center=mitral_centroid_area
)

plotter.add_mesh(
    centroid_points_sphere,
    color="cyan"
)

plotter.add_mesh(
    centroid_area_sphere,
    color="magenta"
)

# apex da massima distanza
apex_dist_sphere = pv.Sphere(
    radius=0.02,
    center=apex_dist_point
)

plotter.add_mesh(
    apex_dist_sphere,
    color="cyan"
)

# apex da PCA
apex_pca_sphere = pv.Sphere(
    radius=0.02,
    center=apex_pca_point
)

plotter.add_mesh(
    apex_pca_sphere,
    color="green"
)

# linea mitrale-apex da massima distanza
line_dist = pv.Line(
    apex_dist_point,
    mitral_centroid
)

plotter.add_mesh(
    line_dist,
    color="yellow",
    line_width=6
)

# linea mitrale-apex da PCA
line_pca = pv.Line(
    apex_pca_point,
    mitral_centroid
)

plotter.add_mesh(
    line_pca,
    color="lime",
    line_width=4
)

# freccia asse massima distanza
arrow_dist = pv.Arrow(
    start=apex_dist_point,
    direction=axis_dist,
    scale=0.3
)

plotter.add_mesh(
    arrow_dist,
    color="yellow"
)

# freccia asse PCA
arrow_pca = pv.Arrow(
    start=apex_pca_point,
    direction=axis_pca,
    scale=0.3
)

plotter.add_mesh(
    arrow_pca,
    color="green"
)

# # linea apex-centroide mitrale
# apex_base_line = pv.Line(
#     apex_point,
#     mitral_centroid
# )

# plotter.add_mesh(
#     apex_base_line,
#     color="yellow",
#     line_width=6
# )

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

# arrow = pv.Arrow(
#     start=apex_point,
#     direction=estimated_apex_base_axis,
#     scale=0.4
# )

# plotter.add_mesh(
#     arrow,
#     color="green"
# )

# ============================================================
# LABELS
# ============================================================

# plotter.add_point_labels(
#     [mitral_centroid],
#     ["Mitral centroid"],
#     font_size=18
# )

# plotter.add_point_labels(
#     [apex_point],
#     ["Apex"],
#     font_size=18
# )

plotter.add_point_labels(
    [mitral_centroid_points],
    ["Vertex centroid"],
    font_size=18
)

plotter.add_point_labels(
    [mitral_centroid_area],
    ["Area centroid"],
    font_size=18
)

plotter.add_point_labels(
    [apex_dist_point],
    ["Apex distance"],
    font_size=18
)

plotter.add_point_labels(
    [apex_pca_point],
    ["Apex PCA"],
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