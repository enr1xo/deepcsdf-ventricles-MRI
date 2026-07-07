"""
Questo codice genera un csv conetente:
- nome del paziente
- posizione del centroide della mitrale di massa (C_area)
- posizione dell'apex stimato con metodo max distance (A_maxD)
- posizione dell'apex stimato con metodo PCA (A_PCA)

Gli apici sono calcolati rispetto al centroide di massa (C_area) che è quello che sembra più stabile tra i due centroidi.
iteriamo su tutti i pazienti in una directory data.
Il csv verrà poi letto in un secondo momento per costruire l'asse apex-base in una delle 4 combinazioni possibili.
"""


from pathlib import Path
import numpy as np
import pandas as pd
import pyvista as pv


ALL_PROCESSED_DIR = Path(
    r"/home/rizzardi/Schreibtisch/AF001_aligned_processed"
)

OUTPUT_CSV = Path(
    r"/home/rizzardi/Schreibtisch/MRI_model/mitral_Carea_apex_MaxD.csv"
)

EXCLUDED_PATIENTS = [
                        "LEU_BBB_21027",
                        "LEU_BBB_21047",
                        "LEU_BBB_21392",
                        "LEU_BBB_21445",
                        "LEU_BBB_21499",
                        "LEU_NORM_2288"
]

APEX_BASE_AXIS = np.array([-1.0, 1.0, 0.0])
APEX_BASE_AXIS /= np.linalg.norm(APEX_BASE_AXIS)

W_PROJ = 0.45
W_AREA = 0.25
W_YX = 0.30


def find_mitral_patch(lv):

    patches = lv.extract_cells(
        lv.cell_data["isholepatch"] == 1
    )

    patches_conn = patches.connectivity()

    region_ids = np.unique(
        patches_conn.cell_data["RegionId"]
    )

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
            "likelihood": likelihood,
        })

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

    return mitral_cells, mitral_region


def compute_area_weighted_centroid(mitral_cells):

    mitral_surf = mitral_cells.extract_surface(algorithm="dataset_surface").triangulate()

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

    if total_area == 0:
        raise ValueError("Mitral patch has zero total area")

    return weighted_sum / total_area


def compute_apex_max_distance(lv_points, C_area):

    distances = np.linalg.norm(
        lv_points - C_area,
        axis=1
    )

    apex_idx = np.argmax(distances)
    apex_point = lv_points[apex_idx]

    return apex_point, apex_idx, distances[apex_idx]


def compute_apex_pca(lv_points, C_area):

    center_lv = lv_points.mean(axis=0)
    X = lv_points - center_lv

    cov = np.cov(X.T)

    eigvals, eigvecs = np.linalg.eigh(cov)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    long_axis = eigvecs[:, 0]

    v_base_to_center = center_lv - C_area

    if np.dot(long_axis, v_base_to_center) < 0:
        long_axis *= -1

    projections = np.dot(
        lv_points - C_area,
        long_axis
    )

    apex_idx = np.argmax(projections)
    apex_point = lv_points[apex_idx]

    return apex_point, apex_idx, projections[apex_idx], long_axis


rows = []

patient_dirs = sorted([
    p for p in ALL_PROCESSED_DIR.iterdir()
    if p.is_dir()
    # if p.name not in EXCLUDED_PATIENTS
])

print(f"Found {len(patient_dirs)} patient folders")

for patient_dir in patient_dirs:

    patient = patient_dir.name

    if patient in EXCLUDED_PATIENTS:
        print(f"skipping {patient}")
        continue

    else:
        lv_path = patient_dir / "lv_endo-processed.vtp"

        print(f"\nProcessing {patient}")

        if not lv_path.exists():
            print(f"  Skipped: missing {lv_path.name}")
            continue

        try:
            lv = pv.read(lv_path)

            mitral_cells, mitral_region = find_mitral_patch(lv)

            C_area = compute_area_weighted_centroid(
                mitral_cells
            )

            lv_points = lv.points

            A_maxD, A_maxD_idx, maxD_value = compute_apex_max_distance(
                lv_points,
                C_area
            )

            # A_PCA, A_PCA_idx, pca_projection, pca_axis = compute_apex_pca(
            #     lv_points,
            #     C_area
            # )

            # apex_distance = np.linalg.norm(
            #     A_maxD - A_PCA
            # )

            rows.append({
                "patient": patient,

                "C_area_x": C_area[0],
                "C_area_y": C_area[1],
                "C_area_z": C_area[2],

                "A_maxD_x": A_maxD[0],
                "A_maxD_y": A_maxD[1],
                "A_maxD_z": A_maxD[2],

                # "A_PCA_x": A_PCA[0],
                # "A_PCA_y": A_PCA[1],
                # "A_PCA_z": A_PCA[2],

                # "apex_distance": apex_distance,
            })

            print("  OK")
            print("  C_area:", C_area)
            print("  A_maxD:", A_maxD)
            # print("  A_PCA:", A_PCA)
            # print("  distance A_maxD - A_PCA:", apex_distance)

        except Exception as e:
            print(f"  Failed: {e}")


df = pd.DataFrame(rows)

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\nDone")
print(f"Processed patients: {len(df)} / {len(patient_dirs)}")
print(f"CSV saved to:\n{OUTPUT_CSV}")