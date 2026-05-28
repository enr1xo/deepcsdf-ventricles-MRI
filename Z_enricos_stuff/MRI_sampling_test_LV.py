import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pyvista as pv

import vtk

from utils.surface_utils import (compute_signed_distance_libigl)
# ============================================================
# PARAMETRI
# ============================================================

PATIENT = "AF001"

ALL_PROCESSED_DIR = Path(
    "/home/rizzardi/Schreibtisch/all_processed_files"
)

MITRAL_CSV = Path(
    "/home/rizzardi/Schreibtisch/subsampling_test/MRI_subsampling/patients_and_mitral_point.csv"
)

# ============================================================
# MRI-LIKE PARAMETERS
# ============================================================

N_SLICES = 8

POINTS_PER_SLICE = 500

SLICE_RADIUS = 0.35

SLICE_SPACING = 0.08

SLICE_THICKNESS = 0.03

# ============================================================
# APEX-BASE AXIS
# ============================================================

axis = np.array(
    [-1.0, 1.0, 0.0]
)

axis /= np.linalg.norm(axis)

# ============================================================
# BUILD ORTHONORMAL BASIS
# ============================================================

tmp = np.array([0.0, 0.0, 1.0])

if abs(np.dot(tmp, axis)) > 0.9:
    tmp = np.array([1.0, 0.0, 0.0])

u = np.cross(axis, tmp)
u /= np.linalg.norm(u)

v = np.cross(axis, u)
v /= np.linalg.norm(v)

# ============================================================
# LOAD MITRAL CSV
# ============================================================

df = pd.read_csv(MITRAL_CSV)

row = df[df["patient"] == PATIENT]

if len(row) == 0:
    raise ValueError(
        f"No mitral point for {PATIENT}"
    )

mitral_point = np.array([
    row.iloc[0]["x"],
    row.iloc[0]["y"],
    row.iloc[0]["z"],
])

print("\nMITRAL POINT")

print(mitral_point)

# ============================================================
# LOAD SURFACES
# ============================================================

patient_dir = (
    ALL_PROCESSED_DIR / PATIENT
)

epi = pv.read(
    patient_dir /
    "epicardium-processed.vtp"
)

lv = pv.read(
    patient_dir /
    "lv_endo-processed.vtp"
)

rv = pv.read(
    patient_dir /
    "rv_endo-processed.vtp"
)


# ============================================================
# STORAGE
# ============================================================

valid_points = []

invalid_points = []

slice_centers = []

# ============================================================
# GENERATE MRI SLICES
# ============================================================

for i in range(N_SLICES):

    # --------------------------------------------------------
    # SLICE CENTER
    # --------------------------------------------------------

    center = (
        mitral_point
        - i * SLICE_SPACING * axis
    )

    slice_centers.append(center)

    # ========================================================
    # GENERATE ALL POINTS OF THE SLICE
    # ========================================================

    slice_points = []

    for _ in range(POINTS_PER_SLICE):

        # ====================================================
        # SAMPLE INSIDE DISC
        # ====================================================

        r = (
            SLICE_RADIUS
            * np.sqrt(np.random.rand())
        )

        theta = (
            2.0
            * np.pi
            * np.random.rand()
        )

        alpha = r * np.cos(theta)

        beta = r * np.sin(theta)

        # ====================================================
        # SLICE THICKNESS
        # ====================================================

        gamma = np.random.uniform(
            -SLICE_THICKNESS / 2.0,
            +SLICE_THICKNESS / 2.0
        )

        # ====================================================
        # BUILD 3D POINT
        # ====================================================

        p = (
            center
            + alpha * u
            + beta  * v
            + gamma * axis
        )

        slice_points.append(p)

    # ========================================================
    # TO NUMPY
    # ========================================================

    slice_points = np.array(
        slice_points
    )

    # ========================================================
    # COMPUTE SDFS IN BATCH
    # ========================================================

    sdf_epi = compute_signed_distance_libigl(
        epi,
        slice_points
    )

    sdf_lv = compute_signed_distance_libigl(
        lv,
        slice_points
    )

    sdf_rv = compute_signed_distance_libigl(
        rv,
        slice_points
    )

    # ========================================================
    # MYOCARDIUM FILTER
    # ========================================================

    invalid = (
        (sdf_epi < 0)
        &
        (sdf_lv > 0)
        &
        (sdf_rv > 0)
    )

    # ========================================================
    # STORE
    # ========================================================

    valid_points.append(
        slice_points[~invalid]
    )

    invalid_points.append(
        slice_points[invalid]
    )

# ============================================================
# CONCATENATE ALL SLICES
# ============================================================

valid_points = np.vstack(
    valid_points
)

invalid_points = np.vstack(
    invalid_points
)

slice_centers = np.array(
    slice_centers
)

print("\n========================================")
print("RESULTS")
print("========================================")

print(
    f"Valid points: "
    f"{len(valid_points)}"
)

print(
    f"Myocardium points: "
    f"{len(invalid_points)}"
)

# ============================================================
# PLOTTER
# ============================================================

plotter = pv.Plotter(
    window_size=(1800, 1400)
)

# ============================================================
# SURFACES
# ============================================================

plotter.add_mesh(
    epi,
    color="lightgray",
    opacity=0.12,
)

plotter.add_mesh(
    lv,
    color="blue",
    opacity=0.08,
)

plotter.add_mesh(
    rv,
    color="green",
    opacity=0.08,
)

# ============================================================
# VALID POINTS
# ============================================================

if len(valid_points) > 0:

    valid_poly = pv.PolyData(
        valid_points
    )

    plotter.add_mesh(
        valid_poly,
        color="lime",
        point_size=5,
        render_points_as_spheres=True,
    )

# ============================================================
# INVALID POINTS
# ============================================================

if len(invalid_points) > 0:

    invalid_poly = pv.PolyData(
        invalid_points
    )

    plotter.add_mesh(
        invalid_poly,
        color="red",
        point_size=5,
        render_points_as_spheres=True,
    )

# ============================================================
# SLICE CENTERS
# ============================================================

slice_centers_poly = pv.PolyData(
    slice_centers
)

plotter.add_mesh(
    slice_centers_poly,
    color="yellow",
    point_size=16,
    render_points_as_spheres=True,
)

# ============================================================
# MITRAL POINT
# ============================================================

plotter.add_mesh(
    pv.PolyData(
        mitral_point.reshape(1, 3)
    ),
    color="magenta",
    point_size=22,
    render_points_as_spheres=True,
)

# ============================================================
# AXIS LINE
# ============================================================

line = pv.Line(
    mitral_point,
    mitral_point - 0.7 * axis,
    resolution=1
)

plotter.add_mesh(
    line,
    color="black",
    line_width=6,
)

# ============================================================
# TEXT
# ============================================================

plotter.add_text(
    f"{PATIENT}\n"
    f"green = valid\n"
    f"red = myocardium",
    font_size=12,
)

# ============================================================
# SHOW
# ============================================================

plotter.show()