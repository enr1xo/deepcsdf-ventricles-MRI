"""
in questo codice iteriamo sui file npy dei pazienti e controlliamo che la prima e la ultima riga (
rispettivamente il primo e l'ultimo punto samplati) abbiano le ultime 3 colonne a [0,0,0].

Ciò significa che almeno il primo e l'ultimo piano sono sopra e sotto la mitrale e l'apex rispettivamente.
"""


"""
Itera sui file .npy dei pazienti e controlla che:

- la prima riga, cioè il primo punto campionato,
- l'ultima riga, cioè l'ultimo punto campionato,

abbiano le ultime tre colonne uguali a [0, 0, 0].

Nel dataset le ultime tre colonne sono:
[mask_epi, mask_lv, mask_rv]

Se sono tutte zero, significa che per quel punto nessuna delle tre
superfici interseca lo slab del relativo piano.

Questo controllo serve a verificare che almeno il primo e l'ultimo
piano siano esterni rispetto alle regioni anatomiche considerate.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

DIR = Path(
    "/home/rizzardi/Schreibtisch/MRI_model"
)

# NPY_DIR = DIR / "generated_npy_incremented_planes" 
NPY_DIR = DIR / "generated_npy" 

OUTPUT_CSV = DIR / "first_last_mask_check_few_planes.csv"


# ============================================================
# PARAMETERS
# ============================================================

EXPECTED_MASK = np.array([0.0, 0.0, 0.0])

# Tolleranza utile nel caso le mask siano salvate come float
ATOL = 1e-8


# ============================================================
# HELPERS
# ============================================================

def extract_patient_name(npy_file: Path) -> str:
    """
    Estrae il nome del paziente dal nome del file.
    Adatta qui i suffissi se necessario.
    """

    name = npy_file.stem

    possible_suffixes = [
        "_three_axis_mri_samples",
        "_mri_samples",
        "_MRI_like_coords_and_sdf",
    ]

    for suffix in possible_suffixes:
        if name.endswith(suffix):
            return name.removesuffix(suffix)

    return name


def mask_is_zero(mask: np.ndarray) -> bool:
    """
    Controlla se una mask è equivalente a [0, 0, 0].
    """

    return bool(
        np.allclose(
            mask,
            EXPECTED_MASK,
            atol=ATOL,
            rtol=0.0,
        )
    )


def check_npy_file(npy_file: Path) -> dict:
    """
    Carica un file .npy e controlla le mask della prima
    e dell'ultima riga.
    """

    patient = extract_patient_name(npy_file)

    try:
        data = np.load(npy_file)

        if data.ndim != 2:
            raise ValueError(
                f"Expected a 2D array, found shape {data.shape}."
            )

        if data.shape[0] == 0:
            raise ValueError("The array contains no rows.")

        if data.shape[1] < 3:
            raise ValueError(
                f"The array has only {data.shape[1]} columns."
            )

        first_mask = data[0, -3:]
        last_mask = data[-1, -3:]

        first_ok = mask_is_zero(first_mask)
        last_ok = mask_is_zero(last_mask)

        return {
            "patient": patient,
            "file": npy_file.name,
            "n_rows": data.shape[0],
            "n_columns": data.shape[1],
            "first_mask_epi": first_mask[0],
            "first_mask_lv": first_mask[1],
            "first_mask_rv": first_mask[2],
            "last_mask_epi": last_mask[0],
            "last_mask_lv": last_mask[1],
            "last_mask_rv": last_mask[2],
            "first_mask_is_zero": first_ok,
            "last_mask_is_zero": last_ok,
            "both_are_zero": first_ok and last_ok,
            "status": "OK" if first_ok and last_ok else "FAILED",
            "error": "",
        }

    except Exception as exc:
        return {
            "patient": patient,
            "file": npy_file.name,
            "n_rows": np.nan,
            "n_columns": np.nan,
            "first_mask_epi": np.nan,
            "first_mask_lv": np.nan,
            "first_mask_rv": np.nan,
            "last_mask_epi": np.nan,
            "last_mask_lv": np.nan,
            "last_mask_rv": np.nan,
            "first_mask_is_zero": False,
            "last_mask_is_zero": False,
            "both_are_zero": False,
            "status": "ERROR",
            "error": str(exc),
        }


# ============================================================
# MAIN
# ============================================================

def main():
    if not NPY_DIR.exists():
        raise FileNotFoundError(
            f"Directory not found: {NPY_DIR}"
        )

    npy_files = sorted(NPY_DIR.glob("*.npy"))

    if not npy_files:
        raise FileNotFoundError(
            f"No .npy files found in: {NPY_DIR}"
        )

    results = []

    for npy_file in npy_files:
        result = check_npy_file(npy_file)
        results.append(result)

        print(
            f"{result['patient']}: "
            f"first={result['first_mask_is_zero']}, "
            f"last={result['last_mask_is_zero']}, "
            f"status={result['status']}"
        )

        if result["status"] == "FAILED":
            print(
                "  First mask:",
                [
                    result["first_mask_epi"],
                    result["first_mask_lv"],
                    result["first_mask_rv"],
                ],
            )

            print(
                "  Last mask:",
                [
                    result["last_mask_epi"],
                    result["last_mask_lv"],
                    result["last_mask_rv"],
                ],
            )

        elif result["status"] == "ERROR":
            print("  Error:", result["error"])

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)

    n_total = len(results_df)
    n_ok = int((results_df["status"] == "OK").sum())
    n_failed = int((results_df["status"] == "FAILED").sum())
    n_errors = int((results_df["status"] == "ERROR").sum())

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files: {n_total}")
    print(f"OK:          {n_ok}")
    print(f"FAILED:      {n_failed}")
    print(f"ERRORS:      {n_errors}")
    print(f"CSV saved:   {OUTPUT_CSV}")

    if n_failed > 0:
        print("\nPatients that failed the check:")

        failed_df = results_df[
            results_df["status"] == "FAILED"
        ]

        for _, row in failed_df.iterrows():
            print(
                f"- {row['patient']}: "
                f"first_zero={row['first_mask_is_zero']}, "
                f"last_zero={row['last_mask_is_zero']}"
            )


if __name__ == "__main__":
    main()