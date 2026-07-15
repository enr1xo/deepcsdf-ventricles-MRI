"""
Questo codice itera su tutte le geometrie processate degli RV e calcola i centroidi della patch tricuspide.

Prima individua la patch tricuspide usando uno scoring basato su:
- proiezione lungo asse apex-base
- area
- posizione rispetto al piano z = 0, incentivando z < 0

Poi calcola il centroide in due modi:
- centroide geometrico
- centroide di massa, pesato per area dei triangoli

Infine salva le distanze tra i due centroidi in un CSV.
"""

from pathlib import Path
import numpy as np
import pyvista as pv
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ALL_PROCESSED_DIR = Path(
    "/home/rizzardi/Schreibtisch/AF001_aligned_processed"
)

output_csv_dir = Path(
    "/home/rizzardi/Schreibtisch/subsampling_test/MRI_subsampling"
)

output_csv_dir.mkdir(parents=True, exist_ok=True)

output_csv_path = output_csv_dir / "tricuspid_centroids_distance_study.csv"

rv_name = "rv_endo-processed.vtp"


# ============================================================
# TRICUSPID INDIVIDUATION
# ============================================================

APEX_BASE_AXIS = np.array([-1.0, 1.0, 0.0])
APEX_BASE_AXIS /= np.linalg.norm(APEX_BASE_AXIS)

# weights
W_PROJ = 0.45
W_AREA = 0.40
W_Z    = 0.15


# ============================================================
# FUNCTIONS
# ============================================================

def find_tricuspid_patch(rv: pv.PolyData) -> tuple[pv.PolyData, dict]:
    """
    Identifica la patch tricuspide tra le hole patches usando uno score basato su:
    - proiezione lungo asse apex-base
    - area
    - prior geometrico rispetto al piano z = 0

    Il prior geometrico incentiva patch con z < 0 usando plane_value = -z.
    """

    if "isholepatch" not in rv.cell_data:
        raise ValueError("Missing cell_data array: 'isholepatch'")

    patches = rv.extract_cells(
        rv.cell_data["isholepatch"] == 1
    )

    if patches.n_cells == 0:
        raise ValueError("No hole patches found: isholepatch == 1 is empty")

    patches_conn = patches.connectivity()

    if "RegionId" not in patches_conn.cell_data:
        raise ValueError("Connectivity did not generate 'RegionId'")

    region_ids = np.unique(
        patches_conn.cell_data["RegionId"]
    )

    patch_infos = []

    for rid in region_ids:

        patch = patches_conn.threshold(
            [rid - 0.5, rid + 0.5],
            scalars="RegionId"
        )

        patch_surf = (
            patch
            .extract_surface(algorithm="dataset_surface")
            .triangulate()
        )

        if patch_surf.n_points == 0 or patch_surf.n_cells == 0:
            continue

        point = patch_surf.points.mean(axis=0)
        x, y, z = point

        projection = np.dot(
            point,
            APEX_BASE_AXIS
        )

        area = patch_surf.area

        # prior geometrico:
        # piano parallelo al piano yx e passante per l'origine => z = 0
        # vogliamo incentivare patch con z < 0
        z_value = -z

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

        z_score = np.clip(
            (z_value + 1.0) / 2.0,
            0.0,
            1.0
        )

        likelihood = (
            W_PROJ * projection_score
            + W_AREA * area_score
            + W_Z * z_score
        )

        patch_infos.append({
            "rid": int(rid),
            "patch": patch_surf,
            "area": area,
            "projection": projection,
            "z": z,
            "z_value": z_value,
            "likelihood": likelihood,
            "n_points": patch_surf.n_points,
            "n_cells": patch_surf.n_cells,
        })

    if len(patch_infos) == 0:
        raise ValueError("No valid connected hole patch found")

    tricuspid_patch_info = max(
        patch_infos,
        key=lambda p: p["likelihood"]
    )

    tricuspid_region = tricuspid_patch_info["rid"]

    labels = np.zeros(
        rv.n_cells,
        dtype=np.int8
    )

    patch_region_ids = patches_conn.cell_data["RegionId"]

    patch_cell_ids = np.where(
        rv.cell_data["isholepatch"] == 1
    )[0]

    tricuspid_mask = (
        patch_region_ids == tricuspid_region
    )

    labels[
        patch_cell_ids[tricuspid_mask]
    ] = 1

    rv.cell_data["tricuspid_patch"] = labels

    tricuspid_cells = rv.extract_cells(
        rv.cell_data["tricuspid_patch"] == 1
    )

    return tricuspid_cells, tricuspid_patch_info


def geometric_centroid(patch: pv.PolyData) -> np.ndarray:
    """
    Centroide geometrico della patch:
    media aritmetica dei punti della patch.
    """

    if patch.n_points == 0:
        raise ValueError("Patch has no points")

    return patch.points.mean(axis=0)


def area_weighted_centroid(patch: pv.PolyData) -> np.ndarray:
    """
    Centroide di massa della patch:
    media dei centroidi triangolari pesata per l'area dei triangoli.
    """

    surf = (
        patch
        .extract_surface(algorithm="dataset_surface")
        .triangulate()
    )

    if surf.n_cells == 0:
        raise ValueError("Patch has no cells")

    faces = surf.faces.reshape(-1, 4)[:, 1:]
    points = surf.points

    weighted_sum = np.zeros(3)
    total_area = 0.0

    for tri in faces:

        p0, p1, p2 = points[tri]

        tri_area = (
            np.linalg.norm(
                np.cross(p1 - p0, p2 - p0)
            ) / 2.0
        )

        tri_centroid = (
            p0 + p1 + p2
        ) / 3.0

        weighted_sum += tri_area * tri_centroid
        total_area += tri_area

    if total_area == 0:
        raise ValueError("Total patch area is zero")

    return weighted_sum / total_area


# ============================================================
# MAIN
# ============================================================

exclude_patients = [
    "LEU_BBB_21027",
    "LEU_BBB_21047",
    "LEU_BBB_21392",
    "LEU_BBB_21445",
    "LEU_BBB_21499",
    "LEU_NORM_2288",
]

rows = []

patient_dirs = sorted([
    p for p in ALL_PROCESSED_DIR.iterdir()
    if p.is_dir()
])

print(f"Found {len(patient_dirs)} patient folders")

for patient_dir in patient_dirs:

    patient_id = patient_dir.name

    if patient_id in exclude_patients:
        print(f"\nSkipping excluded patient: {patient_id}")

        rows.append({
            "patient_id": patient_id,
            "status": "excluded",
        })

        continue

    rv_path = patient_dir / rv_name

    print(f"\nProcessing {patient_id}")

    if not rv_path.exists():
        print(f"  Missing RV file: {rv_path}")

        rows.append({
            "patient_id": patient_id,
            "status": "missing_rv_file",
        })

        continue

    try:
        rv = pv.read(rv_path)

        scale_to_original_range = (
            rv.field_data[
                "scale-tooriginalrange"
            ][0]
        )

        print("original scale factor:", scale_to_original_range)

        tricuspid_cells, tricuspid_info = find_tricuspid_patch(rv)

        tricuspid_surface = (
            tricuspid_cells
            .extract_surface(algorithm="dataset_surface")
            .triangulate()
        )

        c_geom = geometric_centroid(
            tricuspid_surface
        )

        c_mass = area_weighted_centroid(
            tricuspid_surface
        )

        distance = np.linalg.norm(
            c_geom - c_mass
        )

        distance_mm = (
            scale_to_original_range
            * distance
            / 1000.0
        )

        rows.append({
            "patient_id": patient_id,
            "status": "ok",

            "tricuspid_region_id": tricuspid_info["rid"],
            "tricuspid_patch_area": tricuspid_info["area"],
            "tricuspid_patch_projection": tricuspid_info["projection"],
            "tricuspid_patch_z": tricuspid_info["z"],
            "tricuspid_patch_z_value": tricuspid_info["z_value"],
            "tricuspid_patch_likelihood": tricuspid_info["likelihood"],
            "tricuspid_patch_n_points": tricuspid_info["n_points"],
            "tricuspid_patch_n_cells": tricuspid_info["n_cells"],

            "geom_centroid_x": c_geom[0],
            "geom_centroid_y": c_geom[1],
            "geom_centroid_z": c_geom[2],

            "mass_centroid_x": c_mass[0],
            "mass_centroid_y": c_mass[1],
            "mass_centroid_z": c_mass[2],

            "centroid_distance": distance,
            "centroid_distance_mm": distance_mm,
        })

        print(f"  Tricuspid region: {tricuspid_info['rid']}")
        print(f"  Tricuspid z: {tricuspid_info['z']:.6f}")
        print(f"  Geometric centroid: {c_geom}")
        print(f"  Mass centroid:      {c_mass}")
        print(f"  Distance normalized: {distance:.6f}")
        print(f"  Distance mm:         {distance_mm:.6f}")

    except Exception as e:

        print(f"  Error: {e}")

        rows.append({
            "patient_id": patient_id,
            "status": "error",
            "error_message": str(e),
        })


# ============================================================
# CSV GENERATION
# ============================================================

df = pd.DataFrame(rows)

df.to_csv(
    output_csv_path,
    index=False,
    sep=";"
)

print("\nDone.")
print(f"Saved CSV to: {output_csv_path}")

ok_df = df[df["status"] == "ok"]

if len(ok_df) > 0:

    print("\nDistance summary normalized / dimensionless:")
    print(
        ok_df["centroid_distance"].describe()
    )

    print("\nDistance summary in mm:")
    print(
        ok_df["centroid_distance_mm"].describe()
    )