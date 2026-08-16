"""
questo coice prende in input una directory di un file npy e restituisce la media dei samples che abbiamo almeno una maschera diversa da 0.

i punti sono uno per riga ed identificati da 9 colonne:
1->3: coordinate x,y,z
4->6: sdf rispetto epi, lv ed rv
6->9: maschera binaria sulla validità del punto rispetto a epi, lv ed rv

lo lanciamo con:
    MRI_samples_counter.py --npy_dir <path_to_npy_directory>
"""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calcola il numero medio di sample con almeno una maschera "
            "diversa da zero nei file NPY di una directory."
        )
    )

    parser.add_argument(
        "--npy_dir",
        type=Path,
        required=True,
        help="Directory contenente i file .npy",
    )

    args = parser.parse_args()

    npy_dir = args.npy_dir

    if not npy_dir.exists():
        raise FileNotFoundError(f"Directory non trovata: {npy_dir}")

    if not npy_dir.is_dir():
        raise NotADirectoryError(f"Il percorso non è una directory: {npy_dir}")

    npy_files = sorted(npy_dir.glob("*.npy"))

    if len(npy_files) == 0:
        raise FileNotFoundError(
            f"Nessun file .npy trovato nella directory: {npy_dir}"
        )

    valid_samples_per_file = []
    total_samples_per_file = []

    for npy_file in npy_files:

        data = np.load(npy_file)

        if data.ndim != 2 or data.shape[1] != 9:
            print(
                f"ATTENZIONE: salto {npy_file.name}: "
                f"shape {data.shape}, attesa (N, 9)"
            )
            continue

        # Colonne 7, 8, 9 -> indici Python 6, 7, 8
        masks = data[:, 6:9]

        # True se almeno una delle tre maschere è diversa da zero
        valid_points = np.any(masks != 0, axis=1)

        n_valid = np.count_nonzero(valid_points)
        n_total = data.shape[0]

        valid_samples_per_file.append(n_valid)
        total_samples_per_file.append(n_total)

        print(
            f"{npy_file.name}: "
            f"{n_valid}/{n_total} samples con almeno una mask != 0 "
            f"({100 * n_valid / n_total:.2f}%)"
        )

    if len(valid_samples_per_file) == 0:
        raise RuntimeError("Nessun file NPY valido è stato processato.")

    valid_samples_per_file = np.asarray(valid_samples_per_file)
    total_samples_per_file = np.asarray(total_samples_per_file)

    mean_valid = np.mean(valid_samples_per_file)
    mean_total = np.mean(total_samples_per_file)

    print("\n" + "=" * 60)
    print("RISULTATI")
    print("=" * 60)

    print(f"File processati: {len(valid_samples_per_file)}")
    print(f"Media samples totali per file: {mean_total:.2f}")
    print(
        f"Media samples con almeno una mask != 0: "
        f"{mean_valid:.2f}"
    )
    print(
        f"Percentuale media rispetto ai samples totali: "
        f"{100 * mean_valid / mean_total:.2f}%"
    )

    print(f"Minimo: {np.min(valid_samples_per_file)}")
    print(f"Massimo: {np.max(valid_samples_per_file)}")


if __name__ == "__main__":
    main()