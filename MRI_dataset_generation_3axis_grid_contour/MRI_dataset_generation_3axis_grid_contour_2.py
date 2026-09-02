"""
GENERAZIONE PARALLELA DEL DATASET MRI GRID-BASED CON SDF DA CONTOUR
================================================

Questo script elabora più pazienti contemporaneamente usando processi CPU.

Ogni processo:

1. riceve l'identificativo di un paziente;
2. carica autonomamente le superfici;
3. genera i punti e le SDF;
4. salva direttamente il file NPY;
5. restituisce al processo principale solamente un piccolo riepilogo.

Non vengono restituiti al processo principale i grandi array dei campioni,
così si evita di trasferire inutilmente molta memoria fra processi.

La GPU non viene usata da questo script. PyVista/VTK e libigl continuano
a eseguire i calcoli geometrici sulla CPU.
"""

# ora sampliamo lungo i long axis solo in prossimità dei contour.
# inoltre, assrgniamo amschera 1 solo a quei punti samplati lungo il relativo contour, sennà rischiamo di supervisionare ancora punti lonati, che potrebbero essere la causa della confusione

# Queste variabili vanno impostate prima di importare NumPy, libigl
# e il modulo che contiene il generatore.
#
# Impediscono a ogni singolo processo di creare a sua volta molti
# thread interni, causando oversubscription della CPU.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import contextlib
import multiprocessing as mp
import time
import traceback

from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from pathlib import Path

import pandas as pd

from MRI_sampling import (
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
    "MRI_model/generated_npy_three_axis_grid_contourSDF_LAXband2mm"
)


# ============================================================
# PARAMETRI DEL SAMPLING
# ============================================================

PARAMS = GridMRIParams(
    square_spacing_mm=6.0,

    short_axis_slab_width_mm=0.75,
    long_axis_volume_width_mm=0.75,

    n_before_mitral=3,
    n_after_apex=3,

    square_margin_factor=1.5,

    n_points_per_short_axis_plane=1000,
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
    plot_debug=False,
    lax_contour_band_mm=2.0,
    sax_contour_band_mm=2.0, # put None if you don't want to sample in a band around the contour
)


# ============================================================
# PARAMETRI DEL BATCH
# ============================================================

# Parti con 2 o 4.
# Aumenta solamente dopo aver controllato la RAM con htop.
MAX_WORKERS = 12

# False: salta gli NPY già esistenti.
# True: rigenera tutto.
OVERWRITE = False

# None: usa tutti i pazienti disponibili.
#
# Per fare una prova:
#
# PATIENTS_TO_PROCESS = [
#     "AF001",
#     "AF002_P1",
#     "AF002_P2",
#     "AF003",
# ]
PATIENTS_TO_PROCESS = None

# Nasconde i log dettagliati prodotti dal generatore.
SILENCE_PATIENT_LOGS = True

SAVE_ERROR_TRACEBACKS = True


# ============================================================
# UTILITÀ
# ============================================================

def get_patients_from_csv(csv_path):
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


def check_patient_surfaces(patient):
    patient_dir = (
        ALL_PROCESSED_DIR / patient
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

    return missing_files


# ============================================================
# WORKER
# ============================================================

def process_one_patient(patient):
    """
    Funzione eseguita in un processo separato.

    Deve rimanere definita al livello principale del modulo,
    non dentro un'altra funzione, affinché sia serializzabile.
    """

    start_time = time.perf_counter()

    output_npy = (
        OUTPUT_DIR
        / f"{patient}_three_axis_mri_grid_samples.npy"
    )

    # --------------------------------------------------------
    # Output già esistente
    # --------------------------------------------------------

    if output_npy.is_file() and not OVERWRITE:
        return {
            "patient": patient,
            "status": "skipped_existing",
            "n_samples": None,
            "n_planes": None,
            "elapsed_seconds": (
                time.perf_counter() - start_time
            ),
            "output_npy": str(output_npy),
            "error_type": None,
            "error_message": None,
            "traceback": None,
        }

    # --------------------------------------------------------
    # Controllo delle superfici
    # --------------------------------------------------------

    missing_files = check_patient_surfaces(
        patient
    )

    if missing_files:
        return {
            "patient": patient,
            "status": "skipped_missing_surfaces",
            "n_samples": None,
            "n_planes": None,
            "elapsed_seconds": (
                time.perf_counter() - start_time
            ),
            "output_npy": str(output_npy),
            "error_type": None,
            "error_message": (
                "Missing surfaces: "
                + " | ".join(missing_files)
            ),
            "traceback": None,
        }

    # --------------------------------------------------------
    # Generazione
    # --------------------------------------------------------

    try:
        if SILENCE_PATIENT_LOGS:
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull):
                    samples, stats_dataframe, debug = (
                        generate_single_patient_grid_dataset(
                            patient=patient,
                            all_processed_dir=ALL_PROCESSED_DIR,
                            csv_path=LANDMARKS_CSV,
                            output_dir=OUTPUT_DIR,
                            params=PARAMS,
                        )
                    )
        else:
            samples, stats_dataframe, debug = (
                generate_single_patient_grid_dataset(
                    patient=patient,
                    all_processed_dir=ALL_PROCESSED_DIR,
                    csv_path=LANDMARKS_CSV,
                    output_dir=OUTPUT_DIR,
                    params=PARAMS,
                )
            )

        result = {
            "patient": patient,
            "status": "completed",
            "n_samples": int(samples.shape[0]),
            "n_planes": int(
                len(debug["plane_specs"])
            ),
            "elapsed_seconds": (
                time.perf_counter() - start_time
            ),
            "output_npy": str(output_npy),
            "error_type": None,
            "error_message": None,
            "traceback": None,
        }

        # Rilasciamo esplicitamente i riferimenti grandi
        # prima di terminare il worker.
        del samples
        del stats_dataframe
        del debug

        return result

    except Exception as error:
        return {
            "patient": patient,
            "status": "failed",
            "n_samples": None,
            "n_planes": None,
            "elapsed_seconds": (
                time.perf_counter() - start_time
            ),
            "output_npy": str(output_npy),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
        }


# ============================================================
# BATCH PARALLELO
# ============================================================

def generate_parallel_batch():
    if not ALL_PROCESSED_DIR.is_dir():
        raise NotADirectoryError(
            f"Directory non trovata:\n{ALL_PROCESSED_DIR}"
        )

    if not LANDMARKS_CSV.is_file():
        raise FileNotFoundError(
            f"CSV non trovato:\n{LANDMARKS_CSV}"
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

    csv_patients = get_patients_from_csv(
        LANDMARKS_CSV
    )

    surface_directories = {
        path.name
        for path in ALL_PROCESSED_DIR.iterdir()
        if path.is_dir()
    }

    if PATIENTS_TO_PROCESS is None:
        patients = [
            patient
            for patient in csv_patients
            if patient in surface_directories
        ]
    else:
        patients = [
            str(patient)
            for patient in PATIENTS_TO_PROCESS
        ]

    if not patients:
        raise RuntimeError(
            "Nessun paziente da elaborare."
        )

    print("\n" + "=" * 72)
    print("PARALLEL CONTOUR-BASED MRI DATASET GENERATION")
    print("=" * 72)
    print(f"Patients:      {len(patients)}")
    print(f"CPU workers:   {MAX_WORKERS}")
    print(f"Output:        {OUTPUT_DIR}")
    print(f"Overwrite:     {OVERWRITE}")
    print("=" * 72)

    results = []

    completed = 0
    failed = 0
    skipped = 0

    batch_start = time.perf_counter()

    # Il contesto spawn è più prudente quando sono coinvolte
    # librerie native come VTK e libigl.
    multiprocessing_context = mp.get_context(
        "spawn"
    )

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        mp_context=multiprocessing_context,
    ) as executor:

        future_to_patient = {
            executor.submit(
                process_one_patient,
                patient,
            ): patient
            for patient in patients
        }

        for future in as_completed(
            future_to_patient
        ):
            patient = future_to_patient[future]

            try:
                result = future.result()

            except Exception as error:
                result = {
                    "patient": patient,
                    "status": "worker_crashed",
                    "n_samples": None,
                    "n_planes": None,
                    "elapsed_seconds": None,
                    "output_npy": None,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                }

            results.append(result)

            status = result["status"]

            if status == "completed":
                completed += 1

                print(
                    f"[{len(results):4d}/{len(patients)}] "
                    f"{patient} — completed "
                    f"({result['elapsed_seconds']:.1f} s)",
                    flush=True,
                )

            elif status.startswith("skipped"):
                skipped += 1

                print(
                    f"[{len(results):4d}/{len(patients)}] "
                    f"{patient} — skipped",
                    flush=True,
                )

            else:
                failed += 1

                print(
                    f"[{len(results):4d}/{len(patients)}] "
                    f"{patient} — FAILED: "
                    f"{result['error_type']}: "
                    f"{result['error_message']}",
                    flush=True,
                )

                if (
                    SAVE_ERROR_TRACEBACKS
                    and result["traceback"]
                ):
                    traceback_path = (
                        errors_dir
                        / f"{patient}_traceback.txt"
                    )

                    traceback_path.write_text(
                        result["traceback"],
                        encoding="utf-8",
                    )

            # Salvataggio progressivo, così non perdiamo il
            # riepilogo se il processo viene interrotto.
            summary_rows = [
                {
                    key: value
                    for key, value in row.items()
                    if key != "traceback"
                }
                for row in results
            ]

            pd.DataFrame(
                summary_rows
            ).to_csv(
                OUTPUT_DIR
                / "batch_generation_summary.csv",
                index=False,
            )

    elapsed = (
        time.perf_counter() - batch_start
    )

    print("\n" + "=" * 72)
    print("BATCH COMPLETED")
    print("=" * 72)
    print(f"Completed: {completed}")
    print(f"Failed:    {failed}")
    print(f"Skipped:   {skipped}")
    print(f"Elapsed:   {elapsed:.1f} s")
    print("=" * 72)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    generate_parallel_batch()