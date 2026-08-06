from pathlib import Path
import traceback

import pandas as pd

from MRI_dataset_tools_3axis_new_distance import (
    generate_patient_three_axis_mri_dataset,
    ThreeAxisMRIParams,
    find_patient_column,
)


# ============================================================
# PATHS
# ============================================================

ALL_PROCESSED_DIR = Path(
    "/home/rizzardi/Schreibtisch/AF001_aligned_processed"
)

CSV_PATH = Path(
    "/home/rizzardi/Schreibtisch/MRI_model/mitral_apex_tricuspid_locations.csv"
)

OUTPUT_DIR = Path(
    "/home/rizzardi/Schreibtisch/MRI_model/generated_npy_three_axis_new_distance"
)

LOG_CSV_PATH = OUTPUT_DIR / "generation_log.csv"
FAILED_CSV_PATH = OUTPUT_DIR / "generation_failed.csv"


# ============================================================
# PARAMETERS
# ============================================================

params = ThreeAxisMRIParams(
    square_spacing_mm=6.0,
    slab_width_mm=0.75,

    n_before_mitral=3,
    n_after_apex=3,

    square_margin_factor=1.5,

    n_points_per_square=1000,
    min_dist_mm=1.0,

    contour_expansion_mm=25.0,
    batch_size=5000,

    plane_23_shift_mm=25.0,

    save_npy=True,
    save_csv=False,
    plot_debug=False,
)


EXCLUDED_PATIENTS = {
    "LEU_BBB_21027",
    "LEU_BBB_21047",
    "LEU_BBB_21392",
    "LEU_BBB_21445",
    "LEU_BBB_21499",
    "LEU_NORM_2288",
}


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        CSV_PATH,
        sep=";"
    )

    patient_col = find_patient_column(df)

    patients = [
        patient
        for patient in df[patient_col].astype(str)
        if patient not in EXCLUDED_PATIENTS
    ]

    all_stats = []
    failed = []

    print("\n========================================")
    print("Three-axis MRI dataset generation")
    print("Patients found:", len(patients))
    print("Output dir:", OUTPUT_DIR)
    print("========================================")

    for i, patient in enumerate(patients, start=1):

        print("\n----------------------------------------")
        print(f"[{i}/{len(patients)}] Processing {patient}")
        print("----------------------------------------")

        try:
            _, stats, _ = generate_patient_three_axis_mri_dataset(
                patient=patient,
                all_processed_dir=ALL_PROCESSED_DIR,
                csv_path=CSV_PATH,
                output_dir=OUTPUT_DIR,
                params=params,
            )

            all_stats.append(stats)

            print("Done")
            print("Samples:", stats["n_samples"])
            print("Planes:", stats["n_planes"])
            print("Short-axis planes:", stats["n_short_axis_planes"])
            print("Square side mm:", f'{stats["square_side_mm"]:.3f}')
            print("Mask epi:", stats["mask_epi_count"], f'({stats["mask_epi_fraction"]:.2%})')
            print("Mask LV :", stats["mask_lv_count"], f'({stats["mask_lv_fraction"]:.2%})')
            print("Mask RV :", stats["mask_rv_count"], f'({stats["mask_rv_fraction"]:.2%})')
            print("Saved:", stats["out_npy"])

        except Exception as e:

            print("FAILED:", patient)
            print(e)

            failed.append({
                "patient": patient,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

    if all_stats:
        log_df = pd.DataFrame(all_stats)
        log_df.to_csv(
            LOG_CSV_PATH,
            index=False
        )

        print("\nSaved log:", LOG_CSV_PATH)

    if failed:
        failed_df = pd.DataFrame(failed)
        failed_df.to_csv(
            FAILED_CSV_PATH,
            index=False
        )

        print("Saved failed patients:", FAILED_CSV_PATH)

    print("\n========================================")
    print("Generation completed")
    print("Successful:", len(all_stats))
    print("Failed:", len(failed))
    print("========================================")


if __name__ == "__main__":
    main()