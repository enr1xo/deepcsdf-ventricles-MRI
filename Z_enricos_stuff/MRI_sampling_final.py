import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pyvista as pv
from tqdm import tqdm
from loguru import logger

from utils.surface_utils import compute_signed_distance_libigl

#========
# sampling inside myocardium
#=========

sampling_in_myo = True

# ============================================================
# PATHS
# ============================================================

ALL_PROCESSED_DIR = Path(
    "/home/rizzardi/Schreibtisch/AF001_aligned_processed"
)

MITRAL_CSV = Path(
    "/home/rizzardi/Schreibtisch/subsampling_test/MRI_subsampling/patients_and_mitral_point_alignedAF001.csv"
)

OUTPUT_DIR = Path(
    "/home/rizzardi/Schreibtisch/MRI_like_samples_AF001_aligned"
)

if sampling_in_myo:
    OUTPUT_DIR = (OUTPUT_DIR / "myocardium_included")

else:
    OUTPUT_DIR = (OUTPUT_DIR / "myocardium_excluded")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)



# ============================================================
# MRI-LIKE PARAMETERS
# ============================================================

N_SLICES = 22

POINTS_PER_SLICE = 3500

SLICE_RADIUS_MM = 130.0
slice_radius = 1.2

SLICE_SPACING_MM = 6.0

SLICE_THICKNESS_MM = 0.1

# in normalized SDF units
OUTSIDE_BAND = 0.15

MM_TO_UM = 1000.0


# ============================================================
# AXIS
# ============================================================

axis = np.array([-1.0, 1.0, 0.0])
axis /= np.linalg.norm(axis)


# ============================================================
# ORTHONORMAL BASIS
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

mitral_df = pd.read_csv(MITRAL_CSV)


# ============================================================
# PATIENTS
# ============================================================

patient_dirs = sorted([
    p for p in ALL_PROCESSED_DIR.iterdir()
    if p.is_dir()
])

print(f"Found {len(patient_dirs)} patients")


# ============================================================
# LOOP
# ============================================================
failed = []

for patient_dir in tqdm(patient_dirs):

    patient = patient_dir.name

    try:

        out_path = (
            OUTPUT_DIR /
            f"{patient}_MRI_like_coords_and_sdf.npy"
        )

        if out_path.exists():

            print(f"Skipping {patient}")

            continue

        # ====================================================
        # MITRAL POINT
        # ====================================================

        row = mitral_df[
            mitral_df["patient"] == patient
        ]

        if len(row) == 0:

            logger.warning(
                f"No mitral point for {patient}"
            )

            failed.append(patient)

            continue

        mitral_point = np.array([
            row.iloc[0]["x"],
            row.iloc[0]["y"],
            row.iloc[0]["z"],
        ])

        # ====================================================
        # LOAD SURFACES
        # ====================================================

        epi_path = (
            patient_dir /
            "epicardium-processed.vtp"
        )

        lv_path = (
            patient_dir /
            "lv_endo-processed.vtp"
        )

        rv_path = (
            patient_dir /
            "rv_endo-processed.vtp"
        )

        if (
            not epi_path.exists()
            or not lv_path.exists()
            or not rv_path.exists()
        ):

            logger.warning(
                f"Missing processed surfaces for {patient}"
            )

            failed.append(patient)

            continue

        epi = pv.read(epi_path)

        lv = pv.read(lv_path)

        rv = pv.read(rv_path)

        # ====================================================
        # RECOVER ANATOMICAL SCALE
        # ====================================================

        scale_to_original = (
            epi.field_data[
                "scale-tooriginalrange"
            ][0]
        )

        # slice_radius = (
        #     SLICE_RADIUS_MM
        #     * MM_TO_UM
        #     / scale_to_original
        # )

        slice_spacing = (
            SLICE_SPACING_MM
            * MM_TO_UM
            / scale_to_original
        )

        slice_thickness = (
            SLICE_THICKNESS_MM
            * MM_TO_UM
            / scale_to_original
        )

        # ====================================================
        # STORAGE
        # ====================================================

        all_points = []

        all_sdfs = []

        # ====================================================
        # GENERATE MRI-LIKE SLICES
        # ====================================================

        for i in range(N_SLICES):

            # ------------------------------------------------
            # SLICE CENTER
            # ------------------------------------------------

            center = (
                mitral_point
                - i * slice_spacing * axis
            )

            # =================================================
            # SAMPLE INSIDE DISC
            # =================================================

            r = (
                slice_radius
                * np.sqrt(
                    np.random.rand(
                        POINTS_PER_SLICE
                    )
                )
            )

            theta = (
                2.0
                * np.pi
                * np.random.rand(
                    POINTS_PER_SLICE
                )
            )

            alpha = (
                r * np.cos(theta)
            )

            beta = (
                r * np.sin(theta)
            )

            # =================================================
            # THICKNESS ALONG AXIS
            # =================================================

            gamma = np.random.uniform(
                -slice_thickness / 2.0,
                +slice_thickness / 2.0,
                size=POINTS_PER_SLICE,
            )

            # =================================================
            # BUILD 3D POINTS
            # =================================================

            slice_points = (
                center[None, :]
                + alpha[:, None] * u[None, :]
                + beta[:, None] * v[None, :]
                + gamma[:, None] * axis[None, :]
            )

            # =================================================
            # COMPUTE SDFS
            # =================================================

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

            # =================================================
            # ALWAYS REMOVE POINTS TOO FAR OUTSIDE EPICARDIUM
            # =================================================

            near_epi_mask = (
                sdf_epi < OUTSIDE_BAND
            )

            slice_points = (
                slice_points[
                    near_epi_mask
                ]
            )

            sdf_epi = (
                sdf_epi[
                    near_epi_mask
                ]
            )

            sdf_lv = (
                sdf_lv[
                    near_epi_mask
                ]
            )

            sdf_rv = (
                sdf_rv[
                    near_epi_mask
                ]
            )

            # =================================================
            # OPTIONAL MYOCARDIUM FILTER
            # =================================================

            if not sampling_in_myo:

                myocardium_mask = (
                    (sdf_epi < 0)
                    &
                    (sdf_lv > 0)
                    &
                    (sdf_rv > 0)
                )

                keep_mask = (
                    ~myocardium_mask
                )

                slice_points = (
                    slice_points[
                        keep_mask
                    ]
                )

                sdf_epi = (
                    sdf_epi[
                        keep_mask
                    ]
                )

                sdf_lv = (
                    sdf_lv[
                        keep_mask
                    ]
                )

                sdf_rv = (
                    sdf_rv[
                        keep_mask
                    ]
                )

            # =================================================
            # STORE
            # =================================================

            all_points.append(
                slice_points
            )

            all_sdfs.append(
                np.stack(
                    [
                        sdf_epi,
                        sdf_lv,
                        sdf_rv,
                    ],
                    axis=1,
                )
            )

        # ====================================================
        # CONCATENATE
        # ====================================================

        all_points = np.vstack(
            all_points
        )

        all_sdfs = np.vstack(
            all_sdfs
        )

        data = np.hstack([
            all_points,
            all_sdfs,
        ]).astype(np.float32)

        # ====================================================
        # SAVE
        # ====================================================

        np.save(
            out_path,
            data,
            allow_pickle=False,
        )

        print(
            f"{patient}: "
            f"saved {data.shape}"
        )

    except Exception as e:

        logger.error(
            f"Failed {patient}: {e}"
        )

        failed.append(patient)


# ============================================================
# REPORT
# ============================================================

print("\nDONE")
print("Failed patients:")

for p in failed:
    print(p)

print(f"Total failed: {len(failed)}")