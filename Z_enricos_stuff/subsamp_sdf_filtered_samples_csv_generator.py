""""
Questo codice scorre i file npy contententi i punti e le sdf di tutti i pazienti da usare in test.
Applica il filtro sulla magnitudine della sdf (tiene solo quelli con |sdf| < soglia).
Scrive un csv di 5 colonne:
    col1: nome del paziente (lo recueriamo dal nome del file npy)
    col2: numero di punti che hanno superato la soglia sdf rispetto epi
    col3: numero di punti che hanno superato la soglia sdf rispetto lv
    col4: numero di punti che hanno superato la soglia sdf rispetto rv
    col5: sommma delle colonne 2, 3, e 4

serve un parser a cui passiamo:
    -i : directory dove stanno gli npy
    -t : il valore della soglia con cui filtrare
    -o : directory dove salvare il csv



    QUESTO CODICE È FUORVIANTE: per -t troppo alte, i punti samplati, ad esempio su lv, hanno sdf rispetto all'epi comunque minore della soglia, perciò vengon contati più volte!!
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

PATIENTS_TXT_FILE = Path("/home/rizzardi/Schreibtisch/splits/test.txt")

def read_patient_names(txt_file: Path) -> list[str]:
    with open(txt_file, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    return names


def find_patient_npy(input_dir: Path, patient_name: str) -> Path:
    matches = sorted(input_dir.glob(f"{patient_name}*.npy"))

    if len(matches) == 0:
        raise FileNotFoundError(f"No .npy found for patient {patient_name} in {input_dir}")

    if len(matches) > 1:
        raise ValueError(
            f"Multiple .npy files found for patient {patient_name}: "
            + ", ".join(m.name for m in matches)
        )

    return matches[0]


def count_points_below_sdf_threshold(
    data: np.ndarray,
    threshold: float,
) -> tuple[int, int, int, int]:

    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError(f"Expected array with shape (N, 6), got {data.shape}")

    sdf = data[:, 3:]

    # sezioni coerenti col dataset
    epi_slice = slice(0, 1500)
    lv_slice  = slice(1500, 3250)
    rv_slice  = slice(3250, 5000)

    mask_epi = np.abs(sdf[epi_slice, 0]) < threshold
    mask_lv  = np.abs(sdf[lv_slice, 1]) < threshold
    mask_rv  = np.abs(sdf[rv_slice, 2]) < threshold

    n_epi = int(np.sum(mask_epi))
    n_lv  = int(np.sum(mask_lv))
    n_rv  = int(np.sum(mask_rv))

    # totale corretto = somma delle sezioni disgiunte
    n_total = n_epi + n_lv + n_rv

    return n_epi, n_lv, n_rv, n_total


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True, type=Path)
    parser.add_argument("-o", "--output_dir", required=True, type=Path)
    parser.add_argument("-t", "--threshold", type=float, default=0.001)
    parser.add_argument("--csv_name", type=str, default="sdf_threshold_counts.csv")
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if not PATIENTS_TXT_FILE.exists():
        raise FileNotFoundError(f"Patients txt file does not exist: {PATIENTS_TXT_FILE}")

    patient_names = read_patient_names(PATIENTS_TXT_FILE)

    rows = []

    for patient in patient_names:
        npy_file = find_patient_npy(input_dir, patient)
        data = np.load(npy_file)

        n_epi, n_lv, n_rv, n_total = count_points_below_sdf_threshold(
                                    data=data,
                                    threshold=args.threshold,
                                )

        rows.append({
            "patient": patient,
            "n_epi": n_epi,
            "n_lv": n_lv,
            "n_rv": n_rv,
            "n_total": n_epi + n_lv + n_rv,
        })

    df = pd.DataFrame(rows)

    if len(df) != len(patient_names):
        raise RuntimeError(
            f"Mismatch: read {len(patient_names)} patient names, "
            f"but wrote {len(df)} rows to CSV"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.csv_name

    df.to_csv(output_path, index=False)

    print(f"Patients requested: {len(patient_names)}")
    print(f"CSV rows written:   {len(df)}")
    print(f"Saved CSV to: {output_path}")


if __name__ == "__main__":
    main()