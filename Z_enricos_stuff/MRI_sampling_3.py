"""
Costruisce e visualizza quadrati MRI-like lungo l'asse C_area -> A_maxD.

- Legge C_area e A_maxD da CSV
- Carica LV, RV ed epicardio
- Costruisce quadrati ortogonali all'asse apex-base
- Spaziatura controllata da SQUARE_SPACING_MM e scale-tooriginalrange
- Stop: al massimo una slice oltre l'apex
- Due slice prima del centroide
- Due slice dopo l'apex
- Il lato del quadrato è calcolato rispetto alla estensione massima dell'epi nel piano 4 rispetto ai versori u e v
- sampliamo n punti all'interno di ogni quadrato ad una distanza minima reciproca di 1mm
- calcoliamo la sdf: 
    - distanza locale l2 nella slice
    - segno dal winding number rispetto tutta la mesh
- genriamo la maschera binaria della validità dei samples
- plotting
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
import igl


# ============================================================
# PYVISTA SETTINGS
# ============================================================

pv.OFF_SCREEN = False
pv.set_plot_theme("document")
pv.global_theme.jupyter_backend = "none"


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

ALL_PROCESSED_DIR = Path(
    r"/home/rizzardi/Schreibtisch/AF001_aligned_processed"
)

CSV_PATH = Path(
    r"/home/rizzardi/Schreibtisch/MRI_model/mitral_Carea_apex_MaxD.csv"
)

patient = "S73"
patient = "VT001_MUG1"
patient = "yrm3438_v1"
patient = "AF010_P2"
# patient = "AF011"
# patient = "LEU_BBB_21036"
# patient = "LEU_NORM_4590"

patient_dir = ALL_PROCESSED_DIR / patient

lv_path = patient_dir / "lv_endo-processed.vtp"
rv_path = patient_dir / "rv_endo-processed.vtp"
epi_path = patient_dir / "epicardium-processed.vtp"


# ============================================================
# PARAMETERS
# ============================================================

SQUARE_SIDE_LENGTH = 2.0
SQUARE_FRACTION = 0.4

SQUARE_SPACING_MM = 6.0
MM_TO_UM = 1000.0

SHOW_LV = True
SHOW_RV = True
SHOW_EPI = True


margin_factor = 1.25

# ============================================================
# FUNCTIONS
# ============================================================

def find_patient_column(df):
    possible_names = [
        "patient",
        "Patient",
        "PATIENT",
        "patient_id",
        "PatientID",
        "id",
        "ID",
    ]

    for name in possible_names:
        if name in df.columns:
            return name

    raise ValueError(
        "Could not find patient column in CSV. "
        f"Available columns are: {list(df.columns)}"
    )


def read_point_from_row(row, possible_column_sets, point_name):
    for cols in possible_column_sets:
        if all(c in row.index for c in cols):
            return np.array(
                [row[cols[0]], row[cols[1]], row[cols[2]]],
                dtype=float,
            )

    raise ValueError(
        f"Could not find columns for {point_name}. "
        f"Available columns are: {list(row.index)}"
    )


def make_oriented_square_with_point_on_diagonal(
    point,
    normal,
    side_length,
    fraction=0.5,
):
    """
    Crea un quadrato nel piano ortogonale a `normal`.

    `point` non è necessariamente il centro del quadrato:
    viene imposto che `point` stia lungo la diagonale corner0 -> corner2.

    fraction = 0.5  -> point al centro del quadrato
    fraction = 0.4  -> quadrato traslato lungo diagonale
    """

    point = np.asarray(point, dtype=float)
    normal = np.asarray(normal, dtype=float)

    normal_norm = np.linalg.norm(normal)
    if normal_norm == 0:
        raise ValueError("Normal vector has zero norm.")

    normal = normal / normal_norm

    tmp = np.array([1.0, 0.0, 0.0])

    if abs(np.dot(tmp, normal)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])

    u = np.cross(normal, tmp)
    u = u / np.linalg.norm(u)

    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)

    h = side_length / 2.0

    rel_corners = np.array([
        -h * u - h * v,
         h * u - h * v,
         h * u + h * v,
        -h * u + h * v,
    ])

    rel_point_on_diag = (
        (1.0 - fraction) * rel_corners[0]
        + fraction * rel_corners[2]
    )

    square_center = point - rel_point_on_diag

    corners = square_center + rel_corners
    faces = np.hstack([[4, 0, 1, 2, 3]])

    square = pv.PolyData(corners, faces)

    return square, square_center


def make_scaled_squares_until_apex(
    start_point,
    apex_point,
    axis,
    side_length,
    spacing,
    fraction=0.5,
    n_before_start=2,
    n_after_apex=2,
):
    start_point = np.asarray(start_point, dtype=float)
    apex_point = np.asarray(apex_point, dtype=float)
    axis = np.asarray(axis, dtype=float)

    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0:
        raise ValueError("Axis has zero norm.")

    axis = axis / axis_norm

    axis_length = np.linalg.norm(apex_point - start_point)

    if spacing <= 0:
        raise ValueError("Spacing must be positive.")

    n_full_steps = int(np.floor(axis_length / spacing))

    axis_points = []

    # 2 punti prima del centroide
    for i in range(n_before_start, 0, -1):
        axis_points.append(start_point - i * spacing * axis)

    # punti dal centroide fino all'ultimo prima/dentro l'apex
    for i in range(n_full_steps + 1):
        axis_points.append(start_point + i * spacing * axis)

    # punti dopo l'apex
    last_distance = n_full_steps * spacing

    for i in range(1, n_after_apex + 1):
        next_distance = last_distance + i * spacing
        axis_points.append(start_point + next_distance * axis)

    squares = []
    square_centers = []

    for axis_point in axis_points:
        square, square_center = make_oriented_square_with_point_on_diagonal(
            point=axis_point,
            normal=axis,
            side_length=side_length,
            fraction=fraction,
        )

        squares.append(square)
        square_centers.append(square_center)

    return squares, np.asarray(axis_points), np.asarray(square_centers)

def get_square_frame(normal):
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)

    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, normal)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])

    u = np.cross(normal, tmp)
    u /= np.linalg.norm(u)

    v = np.cross(normal, u)
    v /= np.linalg.norm(v)

    return u, v


def estimate_square_side_from_epi_on_slice(
    epi,
    slice_point,
    axis,
    slab_half_width,
    margin_factor=1.1,
):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)

    u, v = get_square_frame(axis)

    epi_points = epi.points

    q = np.dot(epi_points - slice_point, axis)

    slab_mask = np.abs(q) <= slab_half_width
    epi_slice_points = epi_points[slab_mask]

    if len(epi_slice_points) == 0:
        raise ValueError(
            "No epicardial points found in the selected slice slab. "
            "Try increasing slab_half_width."
        )

    local = epi_slice_points - slice_point

    coord_u = np.dot(local, u)
    coord_v = np.dot(local, v)

    range_u = coord_u.max() - coord_u.min()
    range_v = coord_v.max() - coord_v.min()

    side_length = margin_factor * max(abs(range_u), abs(range_v))

    return side_length, epi_slice_points


def estimate_max_points_in_square(side_length, min_dist):
    return int((2.0 / np.sqrt(3.0)) * side_length**2 / min_dist**2)


def sample_points_in_square_min_dist(
    square_center,
    normal,
    side_length,
    n_points,
    min_dist,
    seed=42,
    max_trials=1_000_000,
):
    rng = np.random.default_rng(seed)

    u, v = get_square_frame(normal)
    half = side_length / 2.0

    points_2d = []
    points_3d = []

    n_max_estimated = estimate_max_points_in_square(
        side_length=side_length,
        min_dist=min_dist,
    )

    if n_points > n_max_estimated:
        raise ValueError(
            f"Requested {n_points} points, but only about "
            f"{n_max_estimated} can fit with min_dist={min_dist:.6f}."
        )

    trials = 0

    while len(points_3d) < n_points and trials < max_trials:
        a = rng.uniform(-half, half)
        b = rng.uniform(-half, half)

        candidate_2d = np.array([a, b])

        if len(points_2d) == 0:
            accept = True
        else:
            existing_2d = np.asarray(points_2d)

            dists = np.linalg.norm(
                existing_2d - candidate_2d,
                axis=1,
            )

            accept = np.all(dists >= min_dist)

        if accept:
            candidate_3d = square_center + a * u + b * v

            points_2d.append(candidate_2d)
            points_3d.append(candidate_3d)

        trials += 1

    if len(points_3d) < n_points:
        raise RuntimeError(
            f"Sampling failed: generated only {len(points_3d)} / {n_points} "
            f"after {max_trials} trials."
        )

    return np.asarray(points_3d)

def compute_sign_libigl(mesh, query_points):
    """
    Ritorna solo il segno:
    +1 fuori dalla superficie
    -1 dentro la superficie
    """

    vertices = mesh.points
    faces = mesh.faces.reshape(-1, 4)[:, 1:4].astype(np.int32)

    w = igl.fast_winding_number(
        V=vertices,
        F=faces,
        Q=query_points.astype(np.float64),
    )

    sign = np.sign(0.5 - np.abs(w))

    # controllo orientazione: un punto sicuramente fuori deve avere segno positivo
    bbox_max = mesh.bounds[1::2]

    outside_point = np.array([[
        bbox_max[0] + 100.0,
        bbox_max[1] + 100.0,
        bbox_max[2] + 100.0,
    ]])

    w_out = igl.fast_winding_number(
        V=vertices,
        F=faces,
        Q=outside_point.astype(np.float64),
    )[0]

    outside_sign = np.sign(0.5 - np.abs(w_out))

    if outside_sign < 0:
        sign *= -1

    return sign

def l2_distance_to_surface_in_slab(
    query_points,
    surface_points,
    axis_origin,
    axis,
    slab_half_width,
):
    query_q = np.dot(query_points - axis_origin, axis)
    surf_q = np.dot(surface_points - axis_origin, axis)

    distances = np.full(len(query_points), np.nan, dtype=float)

    for i, p in enumerate(query_points):
        q = query_q[i]

        slab_mask = np.abs(surf_q - q) <= slab_half_width
        slab_points = surface_points[slab_mask]

        if len(slab_points) == 0:
            continue

        dists = np.linalg.norm(slab_points - p[None, :], axis=1)
        distances[i] = np.min(dists)

    return distances


def signed_slab_sdf(
    query_points,
    surface,
    axis_origin,
    axis,
    slab_half_width,
):
    unsigned_distance = l2_distance_to_surface_in_slab(
        query_points=query_points,
        surface_points=surface.points,
        axis_origin=axis_origin,
        axis=axis,
        slab_half_width=slab_half_width,
    )

    sign = compute_sign_libigl(
        mesh=surface,
        query_points=query_points,
    )

    return unsigned_distance * sign

# ============================================================
# LOAD MESHES
# ============================================================

lv = None
rv = None
epi = None

if SHOW_LV:
    lv = pv.read(lv_path)
    print("\nLV loaded")
    print(lv)

if SHOW_RV and rv_path.exists():
    rv = pv.read(rv_path)
    print("\nRV loaded")
    print(rv)

if SHOW_EPI and epi_path.exists():
    epi = pv.read(epi_path)
    print("\nEpi loaded")
    print(epi)

if epi is None:
    raise ValueError("Epicardium mesh is required to read scale-tooriginalrange.")


# ============================================================
# READ C_AREA AND A_MAXD FROM CSV
# ============================================================

df = pd.read_csv(CSV_PATH)

patient_col = find_patient_column(df)

row_df = df[df[patient_col].astype(str) == str(patient)]

if row_df.empty:
    raise ValueError(
        f"Patient '{patient}' not found in CSV.\n"
        f"CSV path: {CSV_PATH}\n"
        f"Patient column: {patient_col}\n"
        f"Available patients example:\n{df[patient_col].head(20).to_list()}"
    )

row = row_df.iloc[0]

mitral_centroid = read_point_from_row(
    row,
    possible_column_sets=[
        ["C_area_x", "C_area_y", "C_area_z"],
        ["Carea_x", "Carea_y", "Carea_z"],
        ["mitral_centroid_x", "mitral_centroid_y", "mitral_centroid_z"],
        ["centroid_x", "centroid_y", "centroid_z"],
        ["C_x", "C_y", "C_z"],
    ],
    point_name="mitral centroid",
)

apex_point = read_point_from_row(
    row,
    possible_column_sets=[
        ["A_maxD_x", "A_maxD_y", "A_maxD_z"],
        ["apex_MaxD_x", "apex_MaxD_y", "apex_MaxD_z"],
        ["apex_maxD_x", "apex_maxD_y", "apex_maxD_z"],
        ["apex_x", "apex_y", "apex_z"],
        ["A_x", "A_y", "A_z"],
    ],
    point_name="apex MaxD",
)

axis = apex_point - mitral_centroid

u, v = get_square_frame(axis)

axis_norm = np.linalg.norm(axis)
if axis_norm == 0:
    raise ValueError("Apex and mitral centroid are identical. Cannot define axis.")

axis = axis / axis_norm

print("\nRead from CSV")
print("Patient:", patient)
print("Mitral centroid C_area:", mitral_centroid)
print("Apex A_maxD:", apex_point)
print("Axis C_area -> A_maxD:", axis)
print("Distance C_area-apex normalized:", np.linalg.norm(apex_point - mitral_centroid))


# ============================================================
# CONVERT SQUARE SPACING FROM MM TO NORMALIZED COORDINATES
# ============================================================

scale_to_original_um = epi.field_data["scale-tooriginalrange"][0]
scale_to_original_mm = scale_to_original_um / 1000.0

SQUARE_SPACING = SQUARE_SPACING_MM / scale_to_original_mm

print("\nScale")
print("scale_to_original:", scale_to_original_um, "um")
print("scale_to_original:", scale_to_original_mm, "mm")
print("square spacing:", SQUARE_SPACING_MM, "mm")
print("square spacing normalized:", SQUARE_SPACING)


SLAB_WIDTH_MM = 1
SLAB_HALF_WIDTH = (SLAB_WIDTH_MM / 2.0) / scale_to_original_mm

REFERENCE_SLICE_INDEX = 4  # n+1 esima slice, perché si parte da 0

reference_slice_point = (
    mitral_centroid
    + REFERENCE_SLICE_INDEX * SQUARE_SPACING * axis
)

SQUARE_SIDE_LENGTH, epi_points_used_for_side = estimate_square_side_from_epi_on_slice(
    epi=epi,
    slice_point=reference_slice_point,
    axis=axis,
    slab_half_width=SLAB_HALF_WIDTH,
    margin_factor=margin_factor,
)

# ============================================================
# CREATE SQUARES UNTIL APEX
# ============================================================

squares, axis_points, square_centers = make_scaled_squares_until_apex(
    start_point=mitral_centroid,
    apex_point=apex_point,
    axis=axis,
    side_length=SQUARE_SIDE_LENGTH,
    spacing=SQUARE_SPACING,
    fraction=SQUARE_FRACTION,
    n_before_start=2,
    n_after_apex=2,
)

print("\nSquares")
print("Number of squares:", len(squares))
print("Number of axis points:", len(axis_points))
print("Number of square centers:", len(square_centers))

axis_distances_norm = np.linalg.norm(axis_points - mitral_centroid, axis=1)
axis_distances_mm = axis_distances_norm * scale_to_original_mm

# print("\nSquare positions along axis [mm]:")
# for i, d in enumerate(axis_distances_mm):
#     flag = " beyond apex" if d > np.linalg.norm(apex_point - mitral_centroid) * scale_to_original_mm else ""
#     print(f"Square {i:02d}: {d:.3f} mm{flag}")

N_SAMPLED_POINTS_PER_SQUARE = 1000
MIN_DIST_MM = 1.0

min_dist_norm = MIN_DIST_MM / scale_to_original_mm

all_sampled_points = []

for i, square_center in enumerate(square_centers):
    sampled_points_i = sample_points_in_square_min_dist(
        square_center=square_center,
        normal=axis,
        side_length=SQUARE_SIDE_LENGTH,
        n_points=N_SAMPLED_POINTS_PER_SQUARE,
        min_dist=min_dist_norm,
        seed=42 + i,
    )

    all_sampled_points.append(sampled_points_i)

all_sampled_points = np.vstack(all_sampled_points)

print("\nSampling")
print("Squares:", len(square_centers))
print("Points per square:", N_SAMPLED_POINTS_PER_SQUARE)
print("Total sampled points:", len(all_sampled_points))
print("Min distance normalized:", min_dist_norm)

#================== sdfs
sdf_epi = signed_slab_sdf(
    query_points=all_sampled_points,
    surface=epi,
    axis_origin=mitral_centroid,
    axis=axis,
    slab_half_width=SLAB_HALF_WIDTH,
)

sdf_lv = signed_slab_sdf(
    query_points=all_sampled_points,
    surface=lv,
    axis_origin=mitral_centroid,
    axis=axis,
    slab_half_width=SLAB_HALF_WIDTH,
)

sdf_rv = signed_slab_sdf(
    query_points=all_sampled_points,
    surface=rv,
    axis_origin=mitral_centroid,
    axis=axis,
    slab_half_width=SLAB_HALF_WIDTH,
)

#---- dataset
mask_epi = np.isfinite(sdf_epi).astype(float)
mask_lv  = np.isfinite(sdf_lv).astype(float)
mask_rv  = np.isfinite(sdf_rv).astype(float)

sdf_epi = np.nan_to_num(sdf_epi, nan=0.0)
sdf_lv  = np.nan_to_num(sdf_lv, nan=0.0)
sdf_rv  = np.nan_to_num(sdf_rv, nan=0.0)

samples = np.column_stack([
    all_sampled_points,
    sdf_epi,
    sdf_lv,
    sdf_rv,
    mask_epi,
    mask_lv,
    mask_rv,
])

# ============================================================
# PLOT
# ============================================================

plotter = pv.Plotter(
    off_screen=False,
    notebook=False,
)

if lv is not None:
    plotter.add_mesh(
        lv,
        color="lightgray",
        opacity=0.48,
    )

if rv is not None:
    plotter.add_mesh(
        rv,
        color="lightblue",
        opacity=0.48,
    )

# if epi is not None:
#     plotter.add_mesh(
#         epi,
#         color="salmon",
#         opacity=0.28,
#     )

plotter.add_mesh(
    pv.Sphere(
        radius=0.02,
        center=mitral_centroid,
    ),
    color="magenta",
)

plotter.add_mesh(
    pv.Sphere(
        radius=0.015,
        center=apex_point,
    ),
    color="yellow",
)

plotter.add_mesh(
    pv.Line(
        mitral_centroid,
        apex_point,
    ),
    color="yellow",
    line_width=7,
)

plotter.add_mesh(
    pv.PolyData(axis_points),
    color="black",
    point_size=10,
    render_points_as_spheres=True,
)

plotter.add_mesh(
    pv.PolyData(square_centers),
    color="magenta",
    point_size=8,
    render_points_as_spheres=True,
)

# ============== quadrati
for square in squares:
    plotter.add_mesh(
        square,
        color="green",
        opacity=0.2,
        show_edges=True,
    )
    break
# #-------------------------

plotter.add_point_labels(
    [mitral_centroid],
    ["C_area"],
    font_size=18,
)

plotter.add_point_labels(
    [apex_point],
    ["A_maxD"],
    font_size=18,
)

# plotter.show_bounds(
#     grid="front",
#     location="outer",
#     all_edges=True,
# )

#========= punti che usiamo per trovare il lato del quadrato
# plotter.add_mesh(
#     pv.PolyData(epi_points_used_for_side),
#     color="blue",
#     point_size=10,
#     render_points_as_spheres=True,
# )

#=================== assi u e v 
ARROW_LENGTH = SQUARE_SIDE_LENGTH / 10

plotter.add_mesh(
    pv.Arrow(
        start=reference_slice_point,
        direction=u,
        scale=ARROW_LENGTH,
    ),
    color="blue",
)

plotter.add_mesh(
    pv.Arrow(
        start=reference_slice_point,
        direction=v,
        scale=ARROW_LENGTH,
    ),
    color="lime",
)

plotter.add_point_labels(
    [
        reference_slice_point + ARROW_LENGTH * u,
        reference_slice_point + ARROW_LENGTH * v,
    ],
    ["u", "v"],
    font_size=18,
)
#-----------------------------------------------------


# ============ sampled points
# plotter.add_mesh(
#     pv.PolyData(all_sampled_points),
#     color="red",
#     point_size=5,
#     render_points_as_spheres=True,
# )
#------------------------------------------------------

#=============== points and sdfs
# sampled_cloud = pv.PolyData(all_sampled_points)

# sampled_cloud["sdf_epi"] = sdf_epi
# abs_max = np.nanmax(sdf_epi)
# abs_min = np.nanmin(sdf_epi)

# sampled_cloud["sdf_lv"] = sdf_lv
# # abs_max = np.nanmin(np.abs(sdf_lv))

# plotter.add_mesh(
#     sampled_cloud,
#     scalars="sdf_epi",
#     cmap="bwr",
#     clim=[abs_min, abs_max],
#     point_size=5,
#     render_points_as_spheres=True,
#     show_scalar_bar=True)
#----------------------------------------------------------------


#=========== points and sdfs with sdf < 0
# mask_inside = sdf_rv < 0

# points_inside = all_sampled_points[mask_inside]
# sdf_inside = sdf_rv[mask_inside]

# inside_cloud = pv.PolyData(points_inside)
# inside_cloud["sdf_rv"] = sdf_inside

# abs_max = np.max(np.abs(sdf_inside))

# plotter.add_mesh(
#     inside_cloud,
#     scalars="sdf_rv",
#     cmap="bwr",
#     clim=[-abs_max, abs_max],
#     point_size=5,
#     render_points_as_spheres=True,
#     show_scalar_bar=True,
# )

# #--------------------


#============== cbar centerd in 0 and nonsymm
# from matplotlib.colors import LinearSegmentedColormap
# sdf = sdf_lv

# sampled_cloud = pv.PolyData(all_sampled_points)
# sampled_cloud[f"{sdf}"] = sdf

# vmin = np.nanmin(sdf)
# vmax = np.nanmax(sdf)

# zero_pos = (0 - vmin) / (vmax - vmin)

# cmap = LinearSegmentedColormap.from_list(
#     "custom_bwr",
#     [
#         (0.0, "blue"),
#         (zero_pos, "white"),
#         (1.0, "red"),
#     ],
# )

# plotter.add_mesh(
#     sampled_cloud,
#     scalars=f"{sdf}",
#     cmap=cmap,
#     clim=[vmin, vmax],
#     point_size=5,
#     render_points_as_spheres=True,
#     show_scalar_bar=True,
# )
#=============0


plotter.add_axes()

plotter.show(
    interactive=True,
    auto_close=False,
)

