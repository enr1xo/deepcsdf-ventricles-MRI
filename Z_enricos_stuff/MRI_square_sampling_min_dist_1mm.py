"""
in questo codice costruiamo il quadrato dove sampliamo andando a leggere la posizione del centroide della mitrale (C_area) e dell'apex (maxD, da C_area) da un csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv

# from scipy.spatial import cKDTree

pv.OFF_SCREEN = False
pv.set_plot_theme("document")
pv.global_theme.jupyter_backend = "none"
# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.surface_utils import compute_signed_distance_libigl

ALL_PROCESSED_DIR = Path(
    r"C:\Users\e.rizzardi\OneDrive\Desktop\processed_patients"
)

CSV_PATH = Path(
    r"C:\Users\e.rizzardi\OneDrive\Desktop\graz_June\mitral_Carea_and_apex_MaxD.csv"
)

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

SQUARE_SIDE_LENGTH = 2.0
SQUARE_DIAGONAL_FRACTION = 0.4

SHOW_RV = True
SHOW_EPI = True


# ============================================================
# FUNCTIONS
# ============================================================

def make_oriented_square(center, normal, side_length):
    """
    Creates a square centered in `center`, lying in the plane normal to `normal`.
    """

    center = np.asarray(center, dtype=float)
    normal = np.asarray(normal, dtype=float)

    normal_norm = np.linalg.norm(normal)
    if normal_norm == 0:
        raise ValueError("Normal vector has zero norm.")

    normal /= normal_norm

    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, normal)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])

    u = np.cross(normal, tmp)
    u /= np.linalg.norm(u)

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

    return pv.PolyData(corners, faces)


def make_oriented_square_with_point_on_diagonal(
    point,
    normal,
    side_length,
    fraction=0.5,
):
    point = np.asarray(point, dtype=float)
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

def find_patient_column(df):
    """
    Tries to find the patient column automatically.
    """

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
    """
    Reads a 3D point from a CSV row, trying different possible column names.
    """

    for cols in possible_column_sets:
        if all(c in row.index for c in cols):
            return np.array([row[cols[0]], row[cols[1]], row[cols[2]]], dtype=float)

    raise ValueError(
        f"Could not find columns for {point_name}. "
        f"Available columns are: {list(row.index)}"
    )

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


def estimate_max_points_in_square(side_length, min_dist):
    """
    Stima realistica del numero massimo di punti nel quadrato
    con distanza minima min_dist.
    """
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

    print("\nSampling feasibility")
    print("Requested points:", n_points)
    print("Estimated max points:", n_max_estimated)

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
            f"after {max_trials} trials. Try increasing max_trials or reducing min_dist."
        )

    return np.asarray(points_3d)

# ============================================================
# LOAD MESHES
# ============================================================

lv = pv.read(lv_path)
print("\nLV loaded")
print(lv)

rv = None
if SHOW_RV and rv_path.exists():
    rv = pv.read(rv_path)
    print("\nRV loaded")
    print(rv)

epi = None
if SHOW_EPI and epi_path.exists():
    epi = pv.read(epi_path)
    print("\nEpi loaded")
    print(epi)


# ============================================================
# READ CENTROID AND APEX FROM CSV
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

axis_norm = np.linalg.norm(axis)
if axis_norm == 0:
    raise ValueError("Apex and mitral centroid are identical. Cannot define axis.")

axis /= axis_norm

print("\nRead from CSV")
print("Patient:", patient)
print("Mitral centroid:", mitral_centroid)
print("Apex MaxD:", apex_point)
print("Axis mitral -> apex:", axis)
print("Distance centroid-apex:", np.linalg.norm(apex_point - mitral_centroid))


# ============================================================
# CREATE SQUARE
# ============================================================

square, square_center = make_oriented_square_with_point_on_diagonal(
    point=mitral_centroid,
    normal=axis,
    side_length=SQUARE_SIDE_LENGTH,
    fraction=SQUARE_DIAGONAL_FRACTION,
)
# ============================================================
# SAMPLE POINTS INSIDE THE SQUARE
# ============================================================

N_SAMPLED_POINTS = 1000
MIN_DIST_MM = 1.0

scale_to_original_um = epi.field_data["scale-tooriginalrange"][0]
scale_to_original_mm = scale_to_original_um / 1000.0

min_dist_norm = MIN_DIST_MM / scale_to_original_mm

print("\nScale")
print("scale_to_original:", scale_to_original_um, "um")
print("scale_to_original:", scale_to_original_mm, "mm")
print("min_dist:", MIN_DIST_MM, "mm")
print("min_dist normalized:", min_dist_norm)

sampled_points = sample_points_in_square_min_dist(
    square_center=square_center,
    normal=axis,
    side_length=SQUARE_SIDE_LENGTH,
    n_points=N_SAMPLED_POINTS,
    min_dist=min_dist_norm,
    seed=42,
)

# ============================================================
# DISTANCE FROM EPICARDIUM IN LOCAL APEX-BASE SLAB
# ============================================================

SLAB_WIDTH_MM = 0.1
SLAB_HALF_WIDTH_NORM = (SLAB_WIDTH_MM / 2.0) / scale_to_original_mm

epi_points = epi.points

# quota lungo asse apex-base
sample_q = np.dot(sampled_points - mitral_centroid, axis)
epi_q = np.dot(epi_points - mitral_centroid, axis)

distances_to_epi = np.full(len(sampled_points), np.nan)

closest_epi_points = np.full_like(sampled_points, np.nan)

for i, (p, q) in enumerate(zip(sampled_points, sample_q)):

    slab_mask = np.abs(epi_q - q) <= SLAB_HALF_WIDTH_NORM

    epi_points_slab = epi_points[slab_mask]

    if len(epi_points_slab) == 0:
        continue

    slab_cloud = pv.PolyData(epi_points_slab)

    idx = slab_cloud.find_closest_point(p)

    closest_point = epi_points_slab[idx]

    dist_norm = np.linalg.norm(p - closest_point)

    distances_to_epi[i] = dist_norm * scale_to_original_mm
    closest_epi_points[i] = closest_point

print("\nDistance to epicardium")
print("Valid distances:", np.sum(~np.isnan(distances_to_epi)), "/", len(distances_to_epi))
print("Mean distance [mm]:", np.nanmean(distances_to_epi))
print("Min distance [mm]:", np.nanmin(distances_to_epi))
print("Max distance [mm]:", np.nanmax(distances_to_epi))


sdf_epi = compute_signed_distance_libigl(epi, sampled_points)

# distanza nella slab con segno dato dalla mesh chiusa epicardica
sign_epi = np.sign(sdf_epi)

signed_slab_distance_mm = distances_to_epi * sign_epi

sampled_cloud = pv.PolyData(sampled_points)
sampled_cloud["SDF"] = sdf_epi

# sampled_cloud = pv.PolyData(sampled_points)
# sampled_cloud = sampled_cloud.compute_implicit_distance(epi)

# sdf_epi = sampled_cloud["implicit_distance"]

# print("SDF min:", sdf_epi.min())
# print("SDF max:", sdf_epi.max())
# print("SDF mean:", sdf_epi.mean())

# ============================================================
# PLOT
# ============================================================

print("PyVista OFF_SCREEN:", pv.OFF_SCREEN)
print("PyVista backend:", pv.global_theme.jupyter_backend)

plotter = pv.Plotter(off_screen=False,
                     notebook=False,)

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

if epi is not None:
    plotter.add_mesh(
        epi,
        color="salmon",
        opacity=0.28,
    )

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

# plotter.add_mesh(
#     square,
#     color="yellow",
#     opacity=0.35,
#     show_edges=True,
# )

plotter.add_point_labels(
    [mitral_centroid],
    ["Mitral centroid"],
    font_size=18,
)

plotter.add_point_labels(
    [apex_point],
    ["Apex MaxD"],
    font_size=18,
)


sampled_points_plt = sampled_points + 1e-3 * axis  # sposto leggermente i punti lungo l'asse per renderli più visibili

# plotter.add_points(
#     sampled_points_plt,
#     color="red",
#     point_size=12,
#     render_points_as_spheres=True,
# )

sampled_points_plot = sampled_points + 5e-3 * axis

sampled_cloud = pv.PolyData(sampled_points_plot)

plotter.add_mesh(
    sampled_cloud,
    color="red",
    point_size=12,
    render_points_as_spheres=True,
)

# for p in sampled_points:
#     plotter.add_mesh(
#         pv.Sphere(radius=0.01, center=p + 5e-3 * axis),
#         color="red",
#     )

sampled_points_plot = sampled_points + 5e-3 * axis

sampled_cloud_plot = pv.PolyData(sampled_points_plot)

sdf_epi = compute_signed_distance_libigl(epi, sampled_points)
sign_epi = np.sign(sdf_epi)

signed_slab_distance_mm = distances_to_epi * sign_epi

sampled_cloud_plot["signed_slab_distance_mm"] = signed_slab_distance_mm

sphere = pv.Sphere(radius=0.008)

sampled_glyphs = sampled_cloud_plot.glyph(
    geom=sphere,
    scale=False,
)

dist_abs_max = np.nanmax(np.abs(signed_slab_distance_mm))

plotter.add_mesh(
    sampled_glyphs,
    scalars="signed_slab_distance_mm",
    cmap="bwr",
    clim=[-dist_abs_max, dist_abs_max],
    show_scalar_bar=True,
)

plotter.show_bounds(
    grid="front",
    location="outer",
    all_edges=True,
)

plotter.add_axes()
plotter.show(
    interactive=True,
    auto_close=False
)

