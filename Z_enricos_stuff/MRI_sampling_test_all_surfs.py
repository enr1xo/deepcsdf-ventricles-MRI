import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pyvista as pv

from utils.surface_utils import (
    compute_signed_distance_libigl
)

# ============================================================
# PARAMETERS
# ============================================================

# PATIENT = "AF004_P1"
# PATIENT = "LEU_BBB_21057"
# PATIENT = "yrm0342_v1"
# PATIENT = "VT001_MUG1"
# PATIENT = "S72"
PATIENT = "AF001"
#========
# sampling inside myocardium
#=========

sampling_in_myo = True

ALL_PROCESSED_DIR = Path(
    "/home/rizzardi/Schreibtisch/AF001_aligned_processed"
)

MITRAL_CSV = Path(
    "/home/rizzardi/Schreibtisch/subsampling_test/MRI_subsampling/patients_and_mitral_point_alignedAF001.csv"
)

# ============================================================
# MRI-LIKE PARAMETERS (REAL mm)
# ============================================================

N_SLICES = 20

POINTS_PER_SLICE = 3500

# disc radius
SLICE_RADIUS_MM = 100.0
SLICE_RADIUS = 1.2

#ellips
ELLIPSE_MAJOR_MM = 60
ELLIPSE_MINOR_MM = 55.0

ELLIPSE_MAJOR = 1.2
ELLIPSE_MINOR = 1.
# distance between slice centers
SLICE_SPACING_MM = 6.0

# slice thickness
SLICE_THICKNESS_MM = 0.1

# outside epicardium band
OUTSIDE_BAND = 0.150

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
# RECOVER ORIGINAL ANATOMICAL SCALE
# ============================================================

scale_to_original = (
    epi.field_data[
        "scale-tooriginalrange"
    ][0]
)

print()
print("Recovered anatomical scale:")
print(scale_to_original)

# ============================================================
# mm -> micron
# ============================================================

MM_TO_UM = 1000.0

# ============================================================
# CONVERT REAL MRI PARAMETERS
# TO NORMALIZED COORDINATES
# ============================================================

# SLICE_RADIUS = (
#     SLICE_RADIUS_MM
#     * MM_TO_UM
#     / scale_to_original
# )


SLICE_SPACING = (
    SLICE_SPACING_MM
    * MM_TO_UM
    / scale_to_original
)

SLICE_THICKNESS = (
    SLICE_THICKNESS_MM
    * MM_TO_UM
    / scale_to_original
)


# ELLIPSE_MAJOR = (
#     ELLIPSE_MAJOR_MM
#     * MM_TO_UM
#     / scale_to_original
# )

# ELLIPSE_MINOR = (
#     ELLIPSE_MINOR_MM
#     * MM_TO_UM
#     / scale_to_original
# )

print()
print("Normalized MRI parameters:")

print("SLICE_RADIUS    =", SLICE_RADIUS)
print("SLICE_SPACING   =", SLICE_SPACING)
print("SLICE_THICKNESS =", SLICE_THICKNESS)
print("OUTSIDE_BAND    =", OUTSIDE_BAND)

# ============================================================
# STORAGE
# ============================================================

lv_points = []

rv_points = []

outside_points = []

myo_points = []

slice_centers = []

# ============================================================
# GENERATE MRI-LIKE SLICES
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
    # GENERATE CANDIDATE POINTS
    # ========================================================

    slice_points = []

    for _ in range(POINTS_PER_SLICE):

        # ====================================================
        # SAMPLE INSIDE DISC
        # ====================================================

        # r = (
        #     SLICE_RADIUS
        #     * np.sqrt(np.random.rand())
        # )

        # theta = (
        #     2.0
        #     * np.pi
        #     * np.random.rand()
        # )

        # alpha = r * np.cos(theta)

        # beta = r * np.sin(theta)

        # ====================================================
        # SAMPLE INSIDE ELLISSE
        # ====================================================
        # SLICE_RADIUS_U = 1.2
        # SLICE_RADIUS_V = 0.7

        # ELLIPSE_A = SLICE_RADIUS_U
        # ELLIPSE_B = SLICE_RADIUS_V

        # r = np.sqrt(np.random.rand())
        # theta = 2.0 * np.pi * np.random.rand()

        # alpha = ELLIPSE_A * r * np.cos(theta)
        # beta = ELLIPSE_B * r * np.sin(theta)

        if ELLIPSE_MAJOR <= ELLIPSE_MINOR:
            raise ValueError("ELLIPSE_MAJOR must be > ELLIPSE_MINOR")

        c_focus = np.sqrt(
            ELLIPSE_MAJOR**2
            - ELLIPSE_MINOR**2
        )

        ellipse_center = (
            center
            - c_focus * u
        )

        r = np.sqrt(
            np.random.rand(POINTS_PER_SLICE)
        )

        theta = (
            2.0
            * np.pi
            * np.random.rand(POINTS_PER_SLICE)
        )

        alpha = (
            ELLIPSE_MAJOR
            * r
            * np.cos(theta)
        )

        beta = (
            ELLIPSE_MINOR
            * r
            * np.sin(theta)
        )

        # ====================================================
        # THICKNESS ALONG AXIS
        # ====================================================

        # gamma = np.random.uniform(
        #     -SLICE_THICKNESS / 2.0,
        #     +SLICE_THICKNESS / 2.0,
            
        # )

        gamma = np.random.uniform(
            -SLICE_THICKNESS / 2.0,
            +SLICE_THICKNESS / 2.0,
            size=POINTS_PER_SLICE,
        )
        # ====================================================
        # BUILD 3D POINT
        # ====================================================

        # p = (
        #     center
        #     + alpha * u
        #     + beta  * v
        #     + gamma * axis
        # )

        # slice_points.append(p)

        # p = (
        #     ellipse_center[None, :]
        #     + alpha[:, None] * u[None, :]
        #     + beta[:, None] * v[None, :]
        #     + gamma[:, None] * axis[None, :]
        # )
        
        slice_points = (
            ellipse_center[None, :]
            + alpha[:, None] * u[None, :]
            + beta[:, None] * v[None, :]
            + gamma[:, None] * axis[None, :]
        )

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
    # REGION MASKS
    # ========================================================

    lv_mask = (
        sdf_lv < 0
    )

    rv_mask = (
        sdf_rv < 0
    )

    outside_mask = (
        (sdf_epi > 0)
        &
        (sdf_epi < OUTSIDE_BAND)
    )

    myocardium_mask = (
        (sdf_epi < 0)
        &
        (sdf_lv > 0)
        &
        (sdf_rv > 0)
    )
    
    # ========================================================
    # STORE
    # ========================================================

    lv_points.append(
        slice_points[lv_mask]
    )

    rv_points.append(
        slice_points[rv_mask]
    )

    outside_points.append(
        slice_points[outside_mask]
    )

    if sampling_in_myo:

        myo_points.append(
            slice_points[
                myocardium_mask
            ]
        )

# ============================================================
# CONCATENATE
# ============================================================

lv_points = np.vstack(
    lv_points
)

rv_points = np.vstack(
    rv_points
)

outside_points = np.vstack(
    outside_points
)

slice_centers = np.array(
    slice_centers
)

if sampling_in_myo:

    myo_points = np.vstack(
        myo_points
    )

else:

    myo_points = np.empty((0, 3))

# ============================================================
# DEBUG
# ============================================================

print("\n========================================")
print("RESULTS")
print("========================================")

print(
    f"LV points: "
    f"{len(lv_points)}"
)

print(
    f"RV points: "
    f"{len(rv_points)}"
)

print(
    f"Myocardium points: "
    f"{len(myo_points)}"
)

print(
    f"Outside points: "
    f"{len(outside_points)}"
)

tot_pts = len(lv_points) + len(rv_points) + len(myo_points) + len(outside_points)
print(
    f"Total points: "
    f"{tot_pts}"
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
# LV POINTS
# ============================================================

if len(lv_points) > 0:

    lv_poly = pv.PolyData(
        lv_points
    )

    plotter.add_mesh(
        lv_poly,
        color="blue",
        point_size=5,
        render_points_as_spheres=True,
    )

# ============================================================
# RV POINTS
# ============================================================

if len(rv_points) > 0:

    rv_poly = pv.PolyData(
        rv_points
    )

    plotter.add_mesh(
        rv_poly,
        color="lime",
        point_size=5,
        render_points_as_spheres=True,
    )

# ============================================================
# OUTSIDE POINTS
# ============================================================

if len(outside_points) > 0:

    outside_poly = pv.PolyData(
        outside_points
    )

    plotter.add_mesh(
        outside_poly,
        color="yellow",
        point_size=5,
        render_points_as_spheres=True,
    )

# ============================================================
# MYOCARDIUM POINTS
# ============================================================

if len(myo_points) > 0:

    myo_poly = pv.PolyData(
        myo_points
    )

    plotter.add_mesh(
        myo_poly,
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
    color="magenta",
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
    color="red",
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
    f"blue   = LV cavity\n"
    f"green  = RV cavity\n"
    f"yellow = outside epi",
    font_size=12,
)

# ============================================================
# SHOW
# ============================================================

plotter.show()