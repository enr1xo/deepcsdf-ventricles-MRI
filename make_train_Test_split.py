#!/usr/bin/env python3

from pathlib import Path
import json


# ============================================================
# PATH
# ============================================================

SPLITS_DIR = Path(
    "/home/rizzardi/Schreibtisch/splits"
)

NPY_DIR = Path(
    # "/home/rizzardi/Schreibtisch/ECHO_model/generated_npy_echo_planes"
    # "/home/rizzardi/Schreibtisch/MRI_model/generated_npy_three_axis"
    # "/home/rizzardi/Schreibtisch/MRI_model/generated_npy_incremented_planes"
    # "/home/rizzardi/Schreibtisch/MRI_model/generated_npy_three_axis_LA_volume_2mm"
    "/home/rizzardi/Schreibtisch/MRI_model/generated_npy_three_axis_grid"
)

TRAIN_TXT = SPLITS_DIR / "train.txt"
TEST_TXT = SPLITS_DIR / "test.txt"

OUTPUT_TRAIN_JSON = SPLITS_DIR / "data_fnames_train.json"
OUTPUT_TEST_JSON = SPLITS_DIR / "data_fnames_test.json"


# ============================================================
# PARAMETRI
# ============================================================

# I file generati dal sampling hanno nomi del tipo:
# AF001_echo_samples.npy
# NPY_SUFFIX = "_echo_samples.npy"
# NPY_SUFFIX = "_mri_samples.npy"
# NPY_SUFFIX = "_three_axis_mri_samples.npy"
NPY_SUFFIX = "_three_axis_mri_grid_samples.npy"


# True:
# salva soltanto il nome del file:
#     "AF001_echo_samples.npy"
#
# False:
# salva il percorso completo:
#     "/home/.../AF001_echo_samples.npy"
SAVE_ONLY_FILENAME = True

# Se True, interrompe lo script quando manca anche un solo NPY.
# Se False, stampa un warning e continua.
FAIL_IF_MISSING = False


# ============================================================
# FUNZIONI
# ============================================================

def read_split_file(split_path: Path) -> list[str]:
    """
    Legge un file train.txt/test.txt.

    Accetta righe come:
        AF001
        AF001.npy
        AF001_echo_samples.npy

    Ignora:
        - righe vuote
        - righe che iniziano con #
    """

    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_path}"
        )

    patient_ids = []

    with split_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            # Se nella riga fosse presente un path,
            # teniamo solamente il nome finale.
            name = Path(line).name

            # Rimuoviamo l'estensione .npy, se presente.
            if name.endswith(".npy"):
                name = name[:-4]

            # Rimuoviamo il suffisso già presente, se necessario.
            suffix_without_extension = NPY_SUFFIX[:-4]

            if name.endswith(suffix_without_extension):
                name = name[:-len(suffix_without_extension)]

            patient_ids.append(name)

    # Elimina duplicati mantenendo l'ordine originale.
    unique_patient_ids = list(dict.fromkeys(patient_ids))

    return unique_patient_ids


def patient_id_to_npy_path(patient_id: str) -> Path:
    """
    Converte:
        AF001
    in:
        NPY_DIR / AF001_echo_samples.npy
    """

    return NPY_DIR / f"{patient_id}{NPY_SUFFIX}"


def create_filename_list(
    patient_ids: list[str],
    split_name: str,
) -> list[str]:

    output_filenames = []
    missing_files = []

    for patient_id in patient_ids:
        npy_path = patient_id_to_npy_path(patient_id)

        if not npy_path.exists():
            missing_files.append(str(npy_path))
            continue

        if SAVE_ONLY_FILENAME:
            output_filenames.append(npy_path.name)
        else:
            output_filenames.append(str(npy_path))

    print()
    print(f"{split_name}")
    print(f"  Patients in split : {len(patient_ids)}")
    print(f"  NPY files found   : {len(output_filenames)}")
    print(f"  NPY files missing : {len(missing_files)}")

    if missing_files:
        print()
        print(f"Missing files in {split_name}:")

        for missing_file in missing_files:
            print(f"  - {missing_file}")

        if FAIL_IF_MISSING:
            raise FileNotFoundError(
                f"{len(missing_files)} NPY files are missing "
                f"for split {split_name}."
            )

    return output_filenames


def save_json(data: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(f"Saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    if not NPY_DIR.exists():
        raise FileNotFoundError(
            f"NPY directory not found: {NPY_DIR}"
        )

    train_patient_ids = read_split_file(TRAIN_TXT)
    test_patient_ids = read_split_file(TEST_TXT)

    train_filenames = create_filename_list(
        patient_ids=train_patient_ids,
        split_name="TRAIN",
    )

    test_filenames = create_filename_list(
        patient_ids=test_patient_ids,
        split_name="TEST",
    )

    save_json(
        data=train_filenames,
        output_path=OUTPUT_TRAIN_JSON,
    )

    save_json(
        data=test_filenames,
        output_path=OUTPUT_TEST_JSON,
    )

    print()
    print("=" * 70)
    print("JSON FILES CREATED")
    print(f"Train samples: {len(train_filenames)}")
    print(f"Test samples : {len(test_filenames)}")
    print(f"Train JSON   : {OUTPUT_TRAIN_JSON}")
    print(f"Test JSON    : {OUTPUT_TEST_JSON}")
    print("=" * 70)


if __name__ == "__main__":
    main()