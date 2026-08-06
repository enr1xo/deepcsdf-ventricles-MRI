"""
GENERAZIONE BATCH DEL DATASET MRI-LIKE GRID-BASED
=================================================

Questo script richiama le funzioni definite in:

    MRI_sampling_GRID_1.py

e genera il file NPY per tutti i pazienti disponibili.

Per ciascun paziente:

1. controlla che siano presenti le tre superfici:
       epicardium-processed.vtp
       lv_endo-processed.vtp
       rv_endo-processed.vtp

2. controlla che il paziente sia presente nel CSV dei landmark;

3. richiama:
       generate_single_patient_grid_dataset(...)

4. salva il file:
       <patient>_three_axis_mri_grid_samples.npy

5. registra l'esito nel file:
       batch_generation_summary.csv

Se un paziente fallisce, l'errore viene registrato e la generazione
continua con il paziente successivo.

Per evitare l'apertura di una finestra PyVista per ogni paziente,
viene impostato:

    plot_debug = False

Per generare effettivamente gli NPY viene impostato:

    save_npy = True
"""

from pathlib import Path
import time
import traceback

import pandas as pd

from MRI_sampling_GRID_1 import (
    GridMRIParams,
    find_patient_column,
    generate_single_patient_grid_dataset,
)


# ============================================================
# PATH
# ============================================================

ALL_PROCESSED_DIR = Path(
    "/home/rizzardi/Schreibtisch/AF001_aligned_processed"
)

LANDMARKS_CSV = Path(
    "/home/rizzardi/Schreibtisch/MRI_model/mitral_apex_tricuspid_locations.csv"
)

OUTPUT_DIR = Path(
    "/home/rizzardi/Schreibtisch/"
    "MRI_model/generated_npy_three_axis_grid"
)


# ============================================================
# PARAMETRI
# ============================================================

PARAMS = GridMRIParams(
    square_spacing_mm=6.0,
    short_axis_slab_width_mm=0.75,
    long_axis_volume_width_mm=2.0,

    n_before_mitral=3,
    n_after_apex=3,

    square_margin_factor=1.5,

    n_points_per_short_axis_plane=500,
    n_points_per_long_axis_volume=1000,

    grid_spacing_mm=1.0,
    contour_expansion_mm=25.0,

    profile_bands_mm=(
        2.0,
        4.0,
        6.0,
        8.0,
        12.0,
    ),

    surface_sampling_fraction=0.80,

    epi_fraction=1.0 / 3.0,
    lv_fraction=1.0 / 3.0,
    rv_fraction=1.0 / 3.0,

    stratification_bins_2d=20,
    stratification_bins_3d=10,

    plane_23_shift_mm=25.0,

    random_grid_offset=True,
    random_seed=42,

    save_npy=True,
    save_csv=False,

    # Fondamentale nel batch:
    # non apre una finestra per ogni paziente.
    plot_debug=False,
)


# ============================================================
# OPZIONI BATCH
# ============================================================

# Se False, un paziente con NPY già presente viene saltato.
# Se True, il file viene rigenerato.
OVERWRITE = False

# Se None, vengono elaborati tutti i pazienti presenti
# contemporaneamente nel CSV e nella directory delle superfici.
#
# Per testarne solo alcuni:
#
# PATIENTS_TO_PROCESS = [
#     "AF001",
#     "AF002_P1",
#     "AF002_P2",
# ]
PATIENTS_TO_PROCESS = None

# Salva anche il traceback completo in un file di testo.
SAVE_ERROR_TRACEBACKS = True


# ============================================================
# FUNZIONI
# ============================================================

def get_patients_from_csv(csv_path: Path) -> list[str]:
    """
    Legge gli ID dei pazienti dal CSV dei landmark.
    """

    dataframe = pd.read_csv(
        csv_path,
        sep=";",
    )

    patient_column = find_patient_column(
        dataframe
    )

    patients = (
        dataframe[patient_column]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    return sorted(patients)


def get_patients_from_surface_directory(
    all_processed_dir: Path,
) -> list[str]:
    """
    Restituisce i nomi delle sottocartelle presenti nella
    directory delle superfici.
    """

    patients = [
        directory.name
        for directory in all_processed_dir.iterdir()
        if directory.is_dir()
    ]

    return sorted(patients)


def check_patient_surfaces(
    patient: str,
    all_processed_dir: Path,
) -> tuple[bool, list[str]]:
    """
    Controlla che siano presenti le tre superfici richieste.
    """

    patient_dir = (
        all_processed_dir / patient
    )

    required_files = [
        patient_dir / "epicardium-processed.vtp",
        patient_dir / "lv_endo-processed.vtp",
        patient_dir / "rv_endo-processed.vtp",
    ]

    missing_files = [
        str(path)
        for path in required_files
        if not path.is_file()
    ]

    return len(missing_files) == 0, missing_files


def generate_batch():
    """
    Genera il dataset per tutti i pazienti selezionati.
    """

    # --------------------------------------------------------
    # Controllo path
    # --------------------------------------------------------

    if not ALL_PROCESSED_DIR.is_dir():
        raise NotADirectoryError(
            "Directory delle superfici non trovata:\n"
            f"{ALL_PROCESSED_DIR}"
        )

    if not LANDMARKS_CSV.is_file():
        raise FileNotFoundError(
            "CSV dei landmark non trovato:\n"
            f"{LANDMARKS_CSV}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    errors_dir = (
        OUTPUT_DIR / "error_tracebacks"
    )

    if SAVE_ERROR_TRACEBACKS:
        errors_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # --------------------------------------------------------
    # Pazienti disponibili
    # --------------------------------------------------------

    csv_patients = set(
        get_patients_from_csv(
            LANDMARKS_CSV
        )
    )

    surface_patients = set(
        get_patients_from_surface_directory(
            ALL_PROCESSED_DIR
        )
    )

    if PATIENTS_TO_PROCESS is None:
        # Elaboriamo solamente pazienti presenti sia nel CSV
        # sia nella directory delle superfici.
        patients = sorted(
            csv_patients & surface_patients
        )
    else:
        patients = [
            str(patient)
            for patient in PATIENTS_TO_PROCESS
        ]

    if not patients:
        raise RuntimeError(
            "Non è stato trovato alcun paziente da elaborare."
        )

    missing_from_csv = sorted(
        surface_patients - csv_patients
    )

    missing_surface_directory = sorted(
        csv_patients - surface_patients
    )

    print("\n" + "=" * 80)
    print("GRID-BASED MRI DATASET GENERATION")
    print("=" * 80)
    print(f"Patients in CSV:              {len(csv_patients)}")
    print(f"Patient directories:          {len(surface_patients)}")
    print(f"Patients selected for batch:  {len(patients)}")
    print(f"Output directory:             {OUTPUT_DIR}")
    print(f"Overwrite existing NPY:       {OVERWRITE}")
    print("=" * 80)

    if missing_from_csv:
        print(
            f"\nWARNING: {len(missing_from_csv)} directories "
            "do not have a corresponding CSV row."
        )

    if missing_surface_directory:
        print(
            f"\nWARNING: {len(missing_surface_directory)} CSV patients "
            "do not have a corresponding surface directory."
        )

    # --------------------------------------------------------
    # Generazione
    # --------------------------------------------------------

    batch_results = []

    n_completed = 0
    n_failed = 0
    n_skipped = 0

    batch_start_time = time.perf_counter()

    for patient_index, patient in enumerate(
        patients,
        start=1,
    ):
        print("\n" + "#" * 80)
        print(
            f"[{patient_index}/{len(patients)}] "
            f"PATIENT: {patient}"
        )
        print("#" * 80)

        patient_start_time = time.perf_counter()

        output_npy = (
            OUTPUT_DIR
            / f"{patient}_three_axis_mri_grid_samples.npy"
        )

        # ----------------------------------------------------
        # File già esistente
        # ----------------------------------------------------

        if output_npy.is_file() and not OVERWRITE:
            elapsed_seconds = (
                time.perf_counter()
                - patient_start_time
            )

            print(
                "SKIPPED: output NPY already exists:\n"
                f"{output_npy}"
            )

            batch_results.append({
                "patient": patient,
                "status": "skipped_existing",
                "n_samples": None,
                "n_planes": None,
                "elapsed_seconds": elapsed_seconds,
                "output_npy": str(output_npy),
                "error_type": None,
                "error_message": None,
            })

            n_skipped += 1
            continue

        # ----------------------------------------------------
        # Controllo CSV
        # ----------------------------------------------------

        if patient not in csv_patients:
            elapsed_seconds = (
                time.perf_counter()
                - patient_start_time
            )

            message = (
                "Patient not present in landmarks CSV."
            )

            print(f"SKIPPED: {message}")

            batch_results.append({
                "patient": patient,
                "status": "skipped_missing_csv",
                "n_samples": None,
                "n_planes": None,
                "elapsed_seconds": elapsed_seconds,
                "output_npy": str(output_npy),
                "error_type": None,
                "error_message": message,
            })

            n_skipped += 1
            continue

        # ----------------------------------------------------
        # Controllo superfici
        # ----------------------------------------------------

        surfaces_available, missing_files = (
            check_patient_surfaces(
                patient=patient,
                all_processed_dir=ALL_PROCESSED_DIR,
            )
        )

        if not surfaces_available:
            elapsed_seconds = (
                time.perf_counter()
                - patient_start_time
            )

            message = (
                "Missing surfaces: "
                + " | ".join(missing_files)
            )

            print(f"SKIPPED: {message}")

            batch_results.append({
                "patient": patient,
                "status": "skipped_missing_surfaces",
                "n_samples": None,
                "n_planes": None,
                "elapsed_seconds": elapsed_seconds,
                "output_npy": str(output_npy),
                "error_type": None,
                "error_message": message,
            })

            n_skipped += 1
            continue

        # ----------------------------------------------------
        # Generazione del paziente
        # ----------------------------------------------------

        try:
            samples, stats_dataframe, debug = (
                generate_single_patient_grid_dataset(
                    patient=patient,
                    all_processed_dir=ALL_PROCESSED_DIR,
                    csv_path=LANDMARKS_CSV,
                    output_dir=OUTPUT_DIR,
                    params=PARAMS,
                )
            )

            elapsed_seconds = (
                time.perf_counter()
                - patient_start_time
            )

            n_samples = int(
                samples.shape[0]
            )

            n_planes = int(
                len(debug["plane_specs"])
            )

            print(
                f"\nCOMPLETED: {patient}"
            )
            print(
                f"Samples: {n_samples}"
            )
            print(
                f"Planes/volumes: {n_planes}"
            )
            print(
                f"Elapsed: {elapsed_seconds:.2f} s"
            )

            batch_results.append({
                "patient": patient,
                "status": "completed",
                "n_samples": n_samples,
                "n_planes": n_planes,
                "elapsed_seconds": elapsed_seconds,
                "output_npy": str(output_npy),
                "error_type": None,
                "error_message": None,
            })

            n_completed += 1

        except Exception as error:
            elapsed_seconds = (
                time.perf_counter()
                - patient_start_time
            )

            error_type = type(error).__name__
            error_message = str(error)

            print(
                f"\nFAILED: {patient}"
            )
            print(
                f"{error_type}: {error_message}"
            )

            batch_results.append({
                "patient": patient,
                "status": "failed",
                "n_samples": None,
                "n_planes": None,
                "elapsed_seconds": elapsed_seconds,
                "output_npy": str(output_npy),
                "error_type": error_type,
                "error_message": error_message,
            })

            if SAVE_ERROR_TRACEBACKS:
                traceback_path = (
                    errors_dir
                    / f"{patient}_traceback.txt"
                )

                with traceback_path.open(
                    "w",
                    encoding="utf-8",
                ) as file:
                    file.write(
                        traceback.format_exc()
                    )

                print(
                    "Traceback saved to:\n"
                    f"{traceback_path}"
                )

            n_failed += 1

        # ----------------------------------------------------
        # Salvataggio progressivo del riepilogo
        # ----------------------------------------------------

        summary_path = (
            OUTPUT_DIR
            / "batch_generation_summary.csv"
        )

        pd.DataFrame(
            batch_results
        ).to_csv(
            summary_path,
            index=False,
        )

        print(
            "\nCurrent batch status:"
        )
        print(
            f"  Completed: {n_completed}"
        )
        print(
            f"  Failed:    {n_failed}"
        )
        print(
            f"  Skipped:   {n_skipped}"
        )

    # --------------------------------------------------------
    # Riepilogo finale
    # --------------------------------------------------------

    total_elapsed_seconds = (
        time.perf_counter()
        - batch_start_time
    )

    summary_dataframe = pd.DataFrame(
        batch_results
    )

    summary_path = (
        OUTPUT_DIR
        / "batch_generation_summary.csv"
    )

    summary_dataframe.to_csv(
        summary_path,
        index=False,
    )

    print("\n" + "=" * 80)
    print("BATCH COMPLETED")
    print("=" * 80)
    print(f"Selected patients: {len(patients)}")
    print(f"Completed:         {n_completed}")
    print(f"Failed:            {n_failed}")
    print(f"Skipped:           {n_skipped}")
    print(
        f"Total elapsed:     "
        f"{total_elapsed_seconds:.2f} s"
    )
    print(
        f"Summary CSV:\n{summary_path}"
    )
    print("=" * 80)

    return summary_dataframe


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    generate_batch()