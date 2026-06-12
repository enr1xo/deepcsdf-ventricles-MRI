"""
Questo codice itera su tutte le geometrie processate degli LV e confronta
i due metodi di individuazione dell'apice.

Per ogni paziente:
1. individua la patch mitralica usando lo scoring area/proiezione/yx;
2. calcola il centroide mitralico area-weighted;
3. trova l'apice con due metodi:
   - massimo della distanza dal centroide mitralico;
   - PCA + massima proiezione lungo il primo asse principale;
4. calcola la distanza tra i due apici;
5. salva i risultati in un CSV.
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

output_csv_path = output_csv_dir / "lv_apex_methods_distance_study.csv"

lv_name = "lv_endo-processed.vtp"


# ============================================================
# MITRAL PATCH PARAMETERS
# ============================================================

APEX_BASE_AXIS = np.array([-1.0, 1.0, 0.0])
APEX_BASE_AXIS /= np.linalg.norm(APEX_BASE_AXIS)

W_PROJ = 0.45
W_AREA = 0.25
W_YX = 0.30


# ============================================================
# FUNCTIONS
# ============================================================

def find_mitral_patch(lv: pv.PolyData) -> tuple[pv.PolyData, dict]:
    """
    Identifica la patch mitrale tra le hole patches usando uno score basato su:
    - proiezione lungo asse apex-base
    - area
    - posizione y+x
    """

    if "isholepatch" not in lv.cell_data:
        raise ValueError("Missing cell_data array: 'isholepatch'")

    patches = lv.extract_cells(
        lv.cell_data["isholepatch"] == 1
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

        point = patch_surf.points[0]
        x, y, z = point

        projection = np.dot(point, APEX_BASE_AXIS)
        area = patch_surf.area
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
            W_PROJ * projection_score
            + W_AREA * area_score
            + W_YX * yx_score
        )

        patch_infos.append({
            "rid": int(rid),
            "area": area,
            "projection": projection,
            "yx_value": yx_value,
            "likelihood": likelihood,
            "n_points": patch_surf.n_points,
            "n_cells": patch_surf.n_cells,
        })

    if len(patch_infos) == 0:
        raise ValueError("No valid connected hole patch found")

    mitral_patch_info = max(
        patch_infos,
        key=lambda p: p["likelihood"]
    )

    mitral_region = mitral_patch_info["rid"]

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

    mitral_cells = lv.extract_cells(
        lv.cell_data["mitral_patch"] == 1
    )

    return mitral_cells, mitral_patch_info


def area_weighted_centroid(patch: pv.PolyData) -> np.ndarray:
    """
    Centroide area-weighted della patch:
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

        tri_area = np.linalg.norm(
            np.cross(p1 - p0, p2 - p0)
        ) / 2.0

        tri_centroid = (p0 + p1 + p2) / 3.0

        weighted_sum += tri_area * tri_centroid
        total_area += tri_area

    if total_area == 0:
        raise ValueError("Total patch area is zero")

    return weighted_sum / total_area


def apex_by_max_distance(
    lv_points: np.ndarray,
    mitral_centroid: np.ndarray
) -> tuple[int, np.ndarray, float, np.ndarray]:
    """
    Metodo 1:
    apice = punto LV più distante dal centroide mitralico.
    """

    distances_from_mitral = np.linalg.norm(
        lv_points - mitral_centroid,
        axis=1
    )

    apex_idx = int(np.argmax(distances_from_mitral))
    apex_point = lv_points[apex_idx]
    apex_distance = distances_from_mitral[apex_idx]

    axis = mitral_centroid - apex_point
    axis /= np.linalg.norm(axis)

    return apex_idx, apex_point, apex_distance, axis


def apex_by_pca_projection(
    lv_points: np.ndarray,
    mitral_centroid: np.ndarray
) -> tuple[int, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Metodo 2:
    - stima asse lungo LV tramite PCA;
    - orienta il primo autovettore in modo coerente;
    - sceglie come apice il punto con massima proiezione lungo tale asse.
    """

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

    apex_idx = int(np.argmax(projections))
    apex_point = lv_points[apex_idx]
    apex_projection = projections[apex_idx]

    axis = mitral_centroid - apex_point
    axis /= np.linalg.norm(axis)

    return (
        apex_idx,
        apex_point,
        apex_projection,
        axis,
        eigvals,
        long_axis,
    )


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

    lv_path = patient_dir / lv_name

    print(f"\nProcessing {patient_id}")

    if not lv_path.exists():
        print(f"  Missing LV file: {lv_path}")

        rows.append({
            "patient_id": patient_id,
            "status": "missing_lv_file",
        })

        continue

    try:
        lv = pv.read(lv_path)

        scale_to_original_range = float(
            lv.field_data["scale-tooriginalrange"][0]
        )

        print("original scale factor:", scale_to_original_range)

        mitral_cells, mitral_info = find_mitral_patch(lv)

        mitral_centroid = area_weighted_centroid(
            mitral_cells
        )

        lv_points = lv.points

        (
            apex_dist_idx,
            apex_dist_point,
            apex_mitral_distance,
            axis_dist,
        ) = apex_by_max_distance(
            lv_points,
            mitral_centroid
        )

        (
            apex_pca_idx,
            apex_pca_point,
            apex_pca_projection,
            axis_pca,
            eigvals,
            long_axis,
        ) = apex_by_pca_projection(
            lv_points,
            mitral_centroid
        )

        apex_methods_distance = np.linalg.norm(
            apex_dist_point - apex_pca_point
        )

        apex_methods_distance_mm = (
            scale_to_original_range
            * apex_methods_distance
            / 1000.0
        )

        rows.append({
            "patient_id": patient_id,
            "status": "ok",

            "scale_to_original_range": scale_to_original_range,

            "mitral_region_id": mitral_info["rid"],
            "mitral_patch_area": mitral_info["area"],
            "mitral_patch_projection": mitral_info["projection"],
            "mitral_patch_yx": mitral_info["yx_value"],
            "mitral_patch_likelihood": mitral_info["likelihood"],
            "mitral_patch_n_points": mitral_info["n_points"],
            "mitral_patch_n_cells": mitral_info["n_cells"],

            "mitral_centroid_x": mitral_centroid[0],
            "mitral_centroid_y": mitral_centroid[1],
            "mitral_centroid_z": mitral_centroid[2],

            "apex_dist_idx": apex_dist_idx,
            "apex_dist_x": apex_dist_point[0],
            "apex_dist_y": apex_dist_point[1],
            "apex_dist_z": apex_dist_point[2],
            "apex_dist_from_mitral": apex_mitral_distance,

            "apex_pca_idx": apex_pca_idx,
            "apex_pca_x": apex_pca_point[0],
            "apex_pca_y": apex_pca_point[1],
            "apex_pca_z": apex_pca_point[2],
            "apex_pca_projection": apex_pca_projection,

            "axis_dist_x": axis_dist[0],
            "axis_dist_y": axis_dist[1],
            "axis_dist_z": axis_dist[2],

            "axis_pca_x": axis_pca[0],
            "axis_pca_y": axis_pca[1],
            "axis_pca_z": axis_pca[2],

            "pca_eigval_1": eigvals[0],
            "pca_eigval_2": eigvals[1],
            "pca_eigval_3": eigvals[2],

            "pca_long_axis_x": long_axis[0],
            "pca_long_axis_y": long_axis[1],
            "pca_long_axis_z": long_axis[2],

            "apex_methods_distance": apex_methods_distance,
            "apex_methods_distance_mm": apex_methods_distance_mm,
        })

        print(f"  Mitral region: {mitral_info['rid']}")
        print(f"  Mitral area-weighted centroid: {mitral_centroid}")

        print(f"  Apex max-distance: idx={apex_dist_idx}, point={apex_dist_point}")
        print(f"  Apex PCA:          idx={apex_pca_idx}, point={apex_pca_point}")

        print(f"  Apex distance normalized: {apex_methods_distance:.6f}")
        print(f"  Apex distance mm:         {apex_methods_distance_mm:.6f}")

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
    print("\nApex distance summary normalized / dimensionless:")
    print(ok_df["apex_methods_distance"].describe())

    print("\nApex distance summary in mm:")
    print(ok_df["apex_methods_distance_mm"].describe())

    print("\nLargest apex differences:")
    print(
        ok_df[
            [
                "patient_id",
                "apex_methods_distance",
                "apex_methods_distance_mm",
            ]
        ]
        .sort_values(
            by="apex_methods_distance_mm",
            ascending=False
        )
        .head(20)
    )