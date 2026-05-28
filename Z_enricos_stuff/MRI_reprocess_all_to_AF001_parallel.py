import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pyvista as pv
from loguru import logger
from tqdm import tqdm

from ventricles_data_preparing_MRI import (
    extract_processed_ventricle_surfaces,
    resolve_patient_dir,
    ensure_vtu_exists,
)

SOURCE_DIR = Path("/home/rizzardi/Schreibtisch/all_processed_files")
SAVE_DIR = Path("/home/rizzardi/Schreibtisch/AF001_aligned_processed")
REFERENCE_PATIENT = "AF001"
MAX_WORKERS = 8

SAVE_DIR.mkdir(parents=True, exist_ok=True)


def get_reference_vtu():
    ref_dir = resolve_patient_dir(SOURCE_DIR, REFERENCE_PATIENT)
    return ensure_vtu_exists(ref_dir)


def process_one_patient(patient, ref_vtu_path):

    patient_save_dir = SAVE_DIR / patient

    epi_out = patient_save_dir / "epicardium-processed.vtp"
    lv_out = patient_save_dir / "lv_endo-processed.vtp"
    rv_out = patient_save_dir / "rv_endo-processed.vtp"

    if epi_out.exists() and lv_out.exists() and rv_out.exists():
        return f"SKIPPED {patient}"

    reference_mesh = pv.read(ref_vtu_path)

    extracted = extract_processed_ventricle_surfaces(
        patient_name=patient,
        reference_name=REFERENCE_PATIENT,
        reference_mesh=reference_mesh,
        source_dir=SOURCE_DIR,
    )

    patient_save_dir.mkdir(parents=True, exist_ok=True)

    extracted["epicardium_surface"].save(epi_out)
    extracted["LV_endo_surface"].save(lv_out)
    extracted["RV_endo_surface"].save(rv_out)
    extracted["original_mesh"].save(patient_save_dir / f"{patient}.vtu")

    return f"SAVED {patient}"


if __name__ == "__main__":

    ref_vtu_path = get_reference_vtu()

    patients = sorted([
        p.name
        for p in SOURCE_DIR.iterdir()
        if p.is_dir()
        and "single_patients_100000pts" not in p.name
    ])

    failed = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = {
            executor.submit(
                process_one_patient,
                patient,
                ref_vtu_path,
            ): patient
            for patient in patients
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
        ):
            patient = futures[future]

            try:
                done_patient = future.result()
                print(f"Saved {done_patient}")

            except Exception as e:
                logger.error(f"Failed {patient}: {e}")
                failed.append(patient)

    print("\nDONE")
    print("Failed patients:")

    for p in failed:
        print(p)

    print(f"Total failed: {len(failed)}")