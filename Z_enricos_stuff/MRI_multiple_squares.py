"""
This script:
1. loads the LV endocardial mesh;
2. identifies the mitral patch among the hole patches;
3. computes the area-weighted mitral centroid;
4. estimates the LV apex using two methods:
   - maximum distance from the mitral centroid;
   - PCA long-axis projection;
5. builds two apex-base axes;
6. plots only:
   - transparent LV mesh;
   - mitral centroid;
   - two apex points;
   - two apex-base axes;
   - two disks centered at the mitral centroid and normal to each axis.
"""

import sys
from pathlib import Path

import numpy as np
import pyvista as pv


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# ALL_PROCESSED_DIR = Path(r"C:\Users\e.rizzardi\OneDrive\Desktop\processed_patients")
ALL_PROCESSED_DIR = Path(r"/home/rizzardi/Schreibtisch/AF001_aligned_processed")

patient = "AF001"
# patient = "yrm0342_v1"
# patient = "LEU_NORM_0016"
# patient = "LEU_BBB_21001"
# patient = "LEU_BBB_21065"
# patient = "LEU_BBB_21350"

patient_dir = ALL_PROCESSED_DIR / patient
lv_path = patient_dir / "lv_endo-processed.vtp"
rv_path = patient_dir / "rv_endo-processed.vtp"
epi_path = patient_dir / "epicardium-processed.vtp"

# ============================================================
# PARAMETERS
# ============================================================

APEX_BASE_AXIS = np.array([-1.0, 1.0, 0.0])
APEX_BASE_AXIS /= np.linalg.norm(APEX_BASE_AXIS)

W_PROJ = 0.45
W_AREA = 0.25
W_YX = 0.30

DISK_RADIUS = 1.0
DISK_RESOLUTION = 100

SQUARE_SIDE_LENGTH = 2

SQUARE_SIDE_LENGTH = 2.0
N_SQUARES = 10
SQUARE_FRACTION = 0.4

SQUARE_SIDE_LENGTH = 2.0
SQUARE_FRACTION = 0.4

SQUARE_SPACING_MM = 6.0
MM_TO_UM = 1000.0

# function to create the squares

def make_oriented_square(center, normal, side_length):
    """
    Creates a square centered in `center`, lying in the plane normal to `normal`.
    """
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)

    # scegli un vettore non parallelo alla normale
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, normal)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])

    # primo asse del piano
    u = np.cross(normal, tmp)
    u /= np.linalg.norm(u)

    # secondo asse del piano
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)

    h = side_length / 2.0

    corners = np.array([
        center - h * u - h * v,
        center + h * u - h * v,
        center + h * u + h * v,
        center - h * u + h * v,
    ])

    faces = np.hstack([[4, 0, 1, 2, 3]])

    square = pv.PolyData(corners, faces)
    return square

def make_oriented_square_with_point_on_diagonal(point, normal, side_length, fraction=0.5):
    """
    Creates a square lying in the plane normal to `normal`.

    `point` lies on one diagonal of the square.
    fraction=0.75 means that `point` is at 3/4 of the diagonal,
    going from corner 0 to corner 2.
    """

    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)

    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, normal)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])

    u = np.cross(normal, tmp)
    u /= np.linalg.norm(u)

    v = np.cross(normal, u)
    v /= np.linalg.norm(v)

    h = side_length / 2.0

    # corners relative to the square center
    rel_corners = np.array([
        -h * u - h * v,
         h * u - h * v,
         h * u + h * v,
        -h * u + h * v,
    ])

    # point is at fraction along diagonal corner 0 -> corner 2
    rel_point_on_diag = (1.0 - fraction) * rel_corners[0] + fraction * rel_corners[2]

    # shift square center so that `point` is exactly there
    square_center = point - rel_point_on_diag

    corners = square_center + rel_corners

    faces = np.hstack([[4, 0, 1, 2, 3]])

    return pv.PolyData(corners, faces)

def make_equally_spaced_squares_along_axis(
    start_point,
    end_point,
    normal,
    side_length,
    n_squares,
    fraction=0.5,
    t_min=0.0,
    t_max=1.0,
):
    """
    Creates n_squares equally spaced squares from start_point to end_point.

    Each square is normal to `normal`.

    fraction:
        0.5 -> axis point is the center of the square
        0.4 -> axis point is shifted along the diagonal
        0.75 -> axis point is at 3/4 of the diagonal
    """

    squares = []

    t_values = np.linspace(t_min, t_max, n_squares)

    for t in t_values:

        point = (1.0 - t) * start_point + t * end_point

        square = make_oriented_square_with_point_on_diagonal(
            point=point,
            normal=normal,
            side_length=side_length,
            fraction=fraction,
        )

        squares.append(square)

    return squares

def make_scaled_squares_until_apex(
    start_point,
    apex_point,
    axis,
    side_length,
    spacing,
    fraction=0.5,
    allow_one_beyond_apex=True,
):
    """
    Generates squares along the apex-base axis.

    Squares start from start_point and move toward apex_point.
    The spacing is in normalized coordinates.

    Stop rule:
    - generate squares up to the apex;
    - allow at most one square beyond the apex.
    """

    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)

    axis_length = np.linalg.norm(
        apex_point - start_point
    )

    n_full_steps = int(
        np.floor(axis_length / spacing)
    )

    centers = []

    for i in range(n_full_steps + 1):

        center = (
            start_point
            + i * spacing * axis
        )

        centers.append(center)

    last_distance = n_full_steps * spacing

    if allow_one_beyond_apex:

        next_distance = last_distance + spacing

        if next_distance > axis_length:

            center = (
                start_point
                + next_distance * axis
            )

            centers.append(center)

    squares = []

    for center in centers:

        square = make_oriented_square_with_point_on_diagonal(
            point=center,
            normal=axis,
            side_length=side_length,
            fraction=fraction,
        )

        squares.append(square)

    return squares, np.array(centers)

# ============================================================
# LOAD surfs
# ============================================================

lv = pv.read(lv_path)

print("\nLV loaded")
print(lv)

rv = pv.read(rv_path)

print("\nRV loaded")
print(rv)

epi = pv.read(epi_path)
print("\nEpi loaded")
print(epi)

# ------ scaling parameter
scale_to_original = epi.field_data["scale-tooriginalrange"][0]

SQUARE_SPACING = (
    SQUARE_SPACING_MM
    * MM_TO_UM
    / scale_to_original
)

print("\nRecovered anatomical scale:")
print(scale_to_original)

print("\nSquare spacing normalized:")
print(SQUARE_SPACING)

# ============================================================
# EXTRACT HOLE PATCHES
# ============================================================

patches = lv.extract_cells(
    lv.cell_data["isholepatch"] == 1
)

patches_conn = patches.connectivity()

region_ids = np.unique(
    patches_conn.cell_data["RegionId"]
)

print("\nFound patch regions:", region_ids)


# ============================================================
# IDENTIFY MITRAL PATCH
# ============================================================

patch_infos = []

for rid in region_ids:

    patch = patches_conn.threshold(
        [rid - 0.5, rid + 0.5],
        scalars="RegionId"
    )

    point = patch.points[0]
    x, y, z = point

    projection = np.dot(point, APEX_BASE_AXIS)
    area = patch.area
    yx_value = y + x

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

    yx_score = np.clip(
        (yx_value + 1.0) / 2.0,
        0.0,
        1.0
    )

    likelihood = (
        W_PROJ * projection_score +
        W_AREA * area_score +
        W_YX * yx_score
    )

    patch_infos.append({
        "rid": rid,
        "patch": patch,
        "point": point,
        "area": area,
        "projection": projection,
        "yx_value": yx_value,
        "likelihood": likelihood,
    })

    print(
        f"Region {rid} | "
        f"area={area:.4f} | "
        f"proj={projection:.4f} | "
        f"yx={yx_value:.4f} | "
        f"LIK={likelihood:.3f}"
    )


mitral_patch_info = max(
    patch_infos,
    key=lambda p: p["likelihood"]
)

mitral_region = mitral_patch_info["rid"]

print("\nChosen mitral region:", mitral_region)


# ============================================================
# CREATE MITRAL LABEL
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
# EXTRACT MITRAL PATCH
# ============================================================

mitral_cells = lv.extract_cells(
    lv.cell_data["mitral_patch"] == 1
)


# ============================================================
# AREA-WEIGHTED MITRAL CENTROID
# ============================================================

mitral_surf = mitral_cells.extract_surface().triangulate()

faces = mitral_surf.faces.reshape(-1, 4)[:, 1:]
points = mitral_surf.points

weighted_sum = np.zeros(3)
total_area = 0.0

for tri in faces:

    p0, p1, p2 = points[tri]

    tri_area = np.linalg.norm(
        np.cross(p1 - p0, p2 - p0)
    ) / 2.0

    tri_centroid = (p0 + p1 + p2) / 3.0

    weighted_sum += tri_area * tri_centroid
    total_area += tri_area

mitral_centroid = weighted_sum / total_area

print("\nArea-weighted mitral centroid:", mitral_centroid)


# ============================================================
# APEX METHOD 1: MAXIMUM DISTANCE
# ============================================================

lv_points = lv.points

distances_from_mitral = np.linalg.norm(
    lv_points - mitral_centroid,
    axis=1
)

apex_dist_idx = np.argmax(distances_from_mitral)
apex_dist_point = lv_points[apex_dist_idx]

axis_dist = apex_dist_point - mitral_centroid
axis_dist /= np.linalg.norm(axis_dist)

print("\nApex by maximum distance")
print("Index:", apex_dist_idx)
print("Point:", apex_dist_point)
print("Distance:", distances_from_mitral[apex_dist_idx])
print("Axis mitral -> apex:", axis_dist)


# ============================================================
# APEX METHOD 2: PCA
# ============================================================

center_lv = lv_points.mean(axis=0)
X = lv_points - center_lv

cov = np.cov(X.T)

eigvals, eigvecs = np.linalg.eigh(cov)

idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

long_axis = eigvecs[:, 0]

v_base_to_center = center_lv - mitral_centroid

if np.dot(long_axis, v_base_to_center) < 0:
    long_axis *= -1

projections = np.dot(
    lv_points - mitral_centroid,
    long_axis
)

apex_pca_idx = np.argmax(projections)
apex_pca_point = lv_points[apex_pca_idx]

axis_pca = apex_pca_point - mitral_centroid
axis_pca /= np.linalg.norm(axis_pca)

print("\nApex by PCA")
print("Eigenvalues:", eigvals)
print("Long axis:", long_axis)
print("Index:", apex_pca_idx)
print("Point:", apex_pca_point)
print("Projection:", projections[apex_pca_idx])
print("Axis mitral -> apex:", axis_pca)

print(
    "\nDistance between apex estimates:",
    np.linalg.norm(apex_dist_point - apex_pca_point)
)


# ============================================================
# DISKS NORMAL TO THE TWO AXES
# ============================================================

disk_dist = pv.Disc(
    center=mitral_centroid,
    inner=0.0,
    outer=DISK_RADIUS,
    normal=axis_dist,
    r_res=1,
    c_res=DISK_RESOLUTION
)

disk_pca = pv.Disc(
    center=mitral_centroid,
    inner=0.0,
    outer=DISK_RADIUS,
    normal=axis_pca,
    r_res=1,
    c_res=DISK_RESOLUTION
)

# third old disk for comparison (the one with axis [-1, 1, 0])
disk_old = pv.Disc(
    center=mitral_centroid,
    inner=0.0,
    outer=DISK_RADIUS,
    normal=np.array([-1.0, 1.0, 0.0]),
    r_res=1,
    c_res=DISK_RESOLUTION
)

# Square for the mitral patch
# ============================================================
# SQUARES NORMAL TO THE TWO AXES
# ============================================================

# square_dist = make_oriented_square(
#     center=mitral_centroid,
#     normal=axis_dist,
#     side_length=SQUARE_SIDE_LENGTH
# )

# square_pca = make_oriented_square(
#     center=mitral_centroid,
#     normal=axis_pca,
#     side_length=SQUARE_SIDE_LENGTH
# )

squares_dist = make_equally_spaced_squares_along_axis(
    start_point=mitral_centroid,
    end_point=apex_dist_point,
    normal=axis_dist,
    side_length=SQUARE_SIDE_LENGTH,
    n_squares=N_SQUARES,
    fraction=SQUARE_FRACTION,
    t_min=0.0,
    t_max=0.95,
)

squares_pca = make_equally_spaced_squares_along_axis(
    start_point=mitral_centroid,
    end_point=apex_pca_point,
    normal=axis_pca,
    side_length=SQUARE_SIDE_LENGTH,
    n_squares=N_SQUARES,
    fraction=SQUARE_FRACTION,
    t_min=0.0,
    t_max=0.95,
)

square_old = make_oriented_square(
    center=mitral_centroid,
    normal=np.array([-1.0, 1.0, 0.0]),
    side_length=SQUARE_SIDE_LENGTH
)

square_dist = make_oriented_square_with_point_on_diagonal(
    point=mitral_centroid,
    normal=axis_dist,
    side_length=SQUARE_SIDE_LENGTH,
    fraction=0.4
)


#  squares physically spaced along the axis_dist, starting from the mitral centroid and moving toward the apex, with a spacing of 6 mm in normalized coordinates
squares_dist, square_centers_dist = make_scaled_squares_until_apex(
    start_point=mitral_centroid,
    apex_point=apex_dist_point,
    axis=axis_dist,
    side_length=SQUARE_SIDE_LENGTH,
    spacing=SQUARE_SPACING,
    fraction=SQUARE_FRACTION,
    allow_one_beyond_apex=True,
)

squares_pca, square_centers_pca = make_scaled_squares_until_apex(
    start_point=mitral_centroid,
    apex_point=apex_pca_point,
    axis=axis_pca,
    side_length=SQUARE_SIDE_LENGTH,
    spacing=SQUARE_SPACING,
    fraction=SQUARE_FRACTION,
    allow_one_beyond_apex=True,
)

print("\nNumber of max-distance squares:", len(squares_dist))
print("Number of PCA squares:", len(squares_pca))

# ============================================================
# CLEAN PLOT
# ============================================================

plotter = pv.Plotter()

# transparent LV mesh
plotter.add_mesh(
    lv,
    color="lightgray",
    opacity=0.48,
)

# transparent RV mesh
plotter.add_mesh(
    rv,
    color="lightblue",
    opacity=0.48,
)

# transparent Epi mesh
plotter.add_mesh(
    epi,
    color="salmon",
    opacity=0.28,
)

# mitral centroid
plotter.add_mesh(
    pv.Sphere(
        radius=0.02,
        center=mitral_centroid
    ),
    color="magenta"
)

# apex from max distance
plotter.add_mesh(
    pv.Sphere(
        radius=0.015,
        center=apex_dist_point
    ),
    color="yellow"
)

# apex from PCA
plotter.add_mesh(
    pv.Sphere(
        radius=0.015,
        center=apex_pca_point
    ),
    color="green"
)

# axis from max distance
plotter.add_mesh(
    pv.Line(
        mitral_centroid,
        apex_dist_point
    ),
    color="yellow",
    line_width=7
)

# axis from PCA
plotter.add_mesh(
    pv.Line(
        mitral_centroid,
        apex_pca_point
    ),
    color="lime",
    line_width=5
)

# ------------------------------------------------------- DISKS
# # disk normal to max-distance axis
# plotter.add_mesh(
#     disk_dist,
#     color="yellow",
#     opacity=0.35,
#     show_edges=True
# )

# # disk normal to PCA axis
# plotter.add_mesh(
#     disk_pca,
#     color="green",
#     opacity=0.35,
#     show_edges=True
# )

# disk normal to old axis
# plotter.add_mesh(
#     disk_old,
#     color="red",
#     opacity=0.35,
#     show_edges=True
# )

# ------------------------------------------------------- SQUARES
# # square normal to max-distance axis
# plotter.add_mesh(
#     square_dist,
#     color="yellow",
#     opacity=0.35,
#     show_edges=True
# )

# # square normal to PCA axis
# plotter.add_mesh(
#     square_pca,
#     color="green",
#     opacity=0.35,
#     show_edges=True
# )

# square normal to old axis
# plotter.add_mesh(
#     square_old,
#     color="red",
#     opacity=0.35,
#     show_edges=True
# )

# for square in squares_dist:
#     plotter.add_mesh(
#         square,
#         color="yellow",
#         opacity=0.25,
#         show_edges=True,
#     )

# for square in squares_dist:

#     plotter.add_mesh(
#         square,
#         color="yellow",
#         opacity=0.12,
#         show_edges=True,
#     )

plotter.add_mesh(
    pv.PolyData(square_centers_dist),
    color="magenta",
    point_size=12,
    render_points_as_spheres=True,
)

for square in squares_pca:

    plotter.add_mesh(
        square,
        color="green",
        opacity=0.12,
        show_edges=True,
    )

# ------------------------ labels


# labels
plotter.add_point_labels(
    [mitral_centroid],
    ["Mitral centroid"],
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

plotter.show_bounds(
    grid="front",
    location="outer",
    all_edges=True,
)

plotter.add_axes()

plotter.show()