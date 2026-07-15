#!/usr/bin/env python3

"""
Ricostruisce i pazienti in gruppi di 16 e visualizza le superfici
ricostruite in una griglia interattiva PyVista 4x4.

Flusso:
1. Legge il JSON di test.
2. Divide i pazienti in gruppi di 16.
3. Per ogni gruppo crea un JSON temporaneo.
4. Richiama test_one_for_parallel.py una sola volta per il gruppo.
5. Carica le mesh ricostruite.
6. Mostra una griglia PyVista 4x4.
7. Alla chiusura del plot passa al gruppo successivo.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pyvista as pv


# ============================================================
# CONFIGURAZIONE
# ============================================================

# Script di test da richiamare
TEST_SCRIPT = Path(
    "/home/rizzardi/workspace/deepcsdf-ventricles-MRI/"
    "Z_enricos_stuff/test_one_for_parallel.py"
)

# Cartella della combinazione
COMBO_DIR = Path(
    "/home/rizzardi/Schreibtisch/MRI_model/MRI_model/3axis"
)

# EXPERIMENT_NAME = "first_experiment_short_only_incremented_planes"
EXPERIMENT_NAME = "eikonal_on"

# "latest" oppure, per esempio, "version_0"
VERSION = "latest"

# Superficie da ricostruire e visualizzare:
# "epicardium", "lv_endo", "rv_endo"
SURFACE = "lv_endo"

# Numero di pazienti visualizzati per finestra
BATCH_SIZE = 16

# Numero di punti usati per il fitting del latent code
NUM_SAMPLES = 512

# Numero di epoche per il fitting del latent code
NUM_EPOCHS = 250

# Modalità di ricostruzione
RECONSTRUCT_FROM = "all"

# Visualizzazione
WINDOW_SIZE = (1800, 1100)
MESH_OPACITY = 1.0
SHOW_EDGES = False

# Se True, tutte le viewport condividono la stessa camera
LINK_CAMERAS = True

# Se True, elimina prima eventuali mesh precedenti dello stesso paziente
# per evitare di caricare risultati vecchi.
DELETE_OLD_RECONSTRUCTIONS = True


# ============================================================
# HELPERS
# ============================================================

def load_module_from_path(module_name: str, module_path: Path):
    """Carica dinamicamente un modulo Python da un percorso."""

    spec = importlib.util.spec_from_file_location(
        module_name,
        str(module_path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Impossibile caricare il modulo: {module_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def get_patient_name(filename: str) -> str:
    """
    Estrae il nome del paziente dal nome del file .npy.
    """

    name = Path(filename).name

    suffixes = [
        "_three_axis_mri_samples.npy",
        "_mri_samples.npy",
        "_MRI_like_coords_and_sdf.npy",
    ]

    for suffix in suffixes:
        if name.endswith(suffix):
            return name.removesuffix(suffix)

    if "-epi_lv_rv_" in name:
        return name.split("-epi_lv_rv_")[0]

    return Path(name).stem


def split_into_batches(items: list, batch_size: int):
    """Divide una lista in gruppi di dimensione batch_size."""

    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def get_latest_version_name() -> str:
    """
    Risolve VERSION="latest" nel nome effettivo, per esempio version_3.
    """

    if VERSION != "latest":
        return VERSION

    experiment_dir = (
        COMBO_DIR
        / "experiments"
        / EXPERIMENT_NAME
    )

    version_dirs = [
        path
        for path in experiment_dir.glob("version_*")
        if path.is_dir()
    ]

    if not version_dirs:
        raise FileNotFoundError(
            f"Nessuna cartella version_* trovata in: {experiment_dir}"
        )

    latest = max(
        version_dirs,
        key=lambda path: int(path.name.split("_")[-1]),
    )

    return latest.name


def find_reconstructed_mesh(
    reconstructed_meshes_dir: Path,
    version_name: str,
    patient: str,
    surface: str,
) -> Path | None:
    """
    Cerca la mesh appena ricostruita.

    Gestisce sia:
        version_0-AF001-epicardium.vtp

    sia:
        version_0-AF001-epicardium-from_la_only.vtp
    """

    if RECONSTRUCT_FROM == "all":
        expected_file = reconstructed_meshes_dir / (
            f"{version_name}-{patient}-{surface}.vtp"
        )
    else:
        expected_file = reconstructed_meshes_dir / (
            f"{version_name}-{patient}-{surface}-from_la_only.vtp"
        )

    if expected_file.exists():
        return expected_file

    # Fallback nel caso il naming sia leggermente diverso
    candidates = sorted(
        reconstructed_meshes_dir.glob(
            f"*{patient}*{surface}*.vtp"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    return candidates[0] if candidates else None


def delete_old_meshes(
    reconstructed_meshes_dir: Path,
    patient_names: list[str],
    surface: str,
):
    """
    Elimina mesh precedenti relative ai pazienti del gruppo corrente.
    """

    for patient in patient_names:
        patterns = [
            f"*-{patient}-{surface}.vtp",
            f"*-{patient}-{surface}-from_la_only.vtp",
        ]

        for pattern in patterns:
            for mesh_file in reconstructed_meshes_dir.glob(pattern):
                print(f"Elimino vecchia mesh: {mesh_file.name}")
                mesh_file.unlink()


# ============================================================
# RICOSTRUZIONE
# ============================================================

def reconstruct_batch(
    batch_entries: list[str],
    batch_index: int,
    temporary_dir: Path,
    reconstructed_meshes_dir: Path,
):
    """
    Crea un JSON temporaneo e richiama lo script di test per il gruppo.
    """

    batch_json = temporary_dir / (
        f"temporary_test_batch_{batch_index:03d}.json"
    )

    with batch_json.open("w", encoding="utf-8") as file:
        json.dump(batch_entries, file, indent=2)

    command = [
        sys.executable,
        str(TEST_SCRIPT),

        "--combo_dir",
        str(COMBO_DIR),

        "--experiment_name",
        EXPERIMENT_NAME,

        "--version",
        VERSION,

        "--override_with_dataset",
        str(batch_json),

        "--surface",
        SURFACE,

        "--reconstruct_from",
        RECONSTRUCT_FROM,

        "--num_samp_per_scene_for_fit",
        str(NUM_SAMPLES),

        "--num_epochs",
        str(NUM_EPOCHS),

        "--save_reconstructed_meshes",

        "--reconstructed_meshes_dir",
        str(reconstructed_meshes_dir),
    ]

    print("\n" + "=" * 80)
    print(f"RICOSTRUZIONE GRUPPO {batch_index + 1}")
    print("=" * 80)
    print("Comando:")
    print(" ".join(command))
    print()

    subprocess.run(
        command,
        check=True,
        cwd=TEST_SCRIPT.parent,
    )

    return batch_json


# ============================================================
# VISUALIZZAZIONE
# ============================================================

def plot_batch(
    patient_names: list[str],
    reconstructed_meshes_dir: Path,
    version_name: str,
    batch_index: int,
    total_batches: int,
):
    """
    Visualizza fino a 16 mesh in una griglia PyVista 4x4.
    """

    plotter = pv.Plotter(
        shape=(4, 4),
        window_size=WINDOW_SIZE,
        title=(
            f"{SURFACE} — gruppo "
            f"{batch_index + 1}/{total_batches}"
        ),
    )

    loaded_meshes = []

    for subplot_index in range(16):
        row = subplot_index // 4
        column = subplot_index % 4

        plotter.subplot(row, column)

        if subplot_index >= len(patient_names):
            plotter.set_background("white")
            continue

        patient = patient_names[subplot_index]

        mesh_file = find_reconstructed_mesh(
            reconstructed_meshes_dir=reconstructed_meshes_dir,
            version_name=version_name,
            patient=patient,
            surface=SURFACE,
        )

        plotter.add_text(
            patient,
            position="upper_left",
            font_size=10,
        )

        if mesh_file is None:
            plotter.add_text(
                "MESH NON TROVATA",
                position="lower_left",
                font_size=9,
            )
            plotter.set_background("white")
            continue

        try:
            mesh = pv.read(mesh_file)

            if mesh.n_points == 0:
                raise ValueError("Mesh senza punti.")

            loaded_meshes.append(mesh)

            plotter.add_mesh(
                mesh,
                opacity=MESH_OPACITY,
                show_edges=SHOW_EDGES,
                smooth_shading=True,
            )

            plotter.add_text(
                f"{mesh.n_points} punti",
                position="lower_left",
                font_size=8,
            )

            plotter.set_background("white")
            plotter.view_isometric()
            plotter.reset_camera()

        except Exception as exc:
            plotter.add_text(
                f"ERRORE:\n{exc}",
                position="lower_left",
                font_size=8,
            )
            plotter.set_background("white")

    if LINK_CAMERAS:
        plotter.link_views()

    plotter.add_text(
        (
            f"{SURFACE} | gruppo {batch_index + 1}/{total_batches}\n"
            "Chiudi la finestra per procedere con il gruppo successivo."
        ),
        position="upper_edge",
        font_size=11,
    )

    print(
        f"\nVisualizzazione gruppo {batch_index + 1}/{total_batches}."
    )
    print(
        "Chiudi la finestra PyVista per continuare con il gruppo successivo."
    )

    plotter.show()

    # Mantiene riferimenti alle mesh fino alla chiusura del plot
    del loaded_meshes


# ============================================================
# MAIN
# ============================================================

def main():
    if not TEST_SCRIPT.exists():
        raise FileNotFoundError(
            f"Script di test non trovato: {TEST_SCRIPT}"
        )

    if not COMBO_DIR.exists():
        raise FileNotFoundError(
            f"Combination directory non trovata: {COMBO_DIR}"
        )

    config_path = COMBO_DIR / "config.py"

    if not config_path.exists():
        raise FileNotFoundError(
            f"config.py non trovato: {config_path}"
        )

    combo_config = load_module_from_path(
        "combo_config_batch_visualization",
        config_path,
    )

    temporary_meshes_dir = (
        COMBO_DIR / "temporary_reconstructed_meshes"
    )

    temporary_meshes_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reconstructed_meshes_dir = temporary_meshes_dir

    test_json = (
        COMBO_DIR
        / "test"
        / "data_fnames_test.json"
    )

    if not test_json.exists():
        raise FileNotFoundError(
            f"JSON di test non trovato: {test_json}"
        )

    with test_json.open("r", encoding="utf-8") as file:
        test_entries = json.load(file)

    if not isinstance(test_entries, list):
        raise ValueError(
            "Il JSON di test deve contenere una lista di file."
        )

    if len(test_entries) == 0:
        raise ValueError("Il JSON di test è vuoto.")

    version_name = get_latest_version_name()

    temporary_dir = (
        COMBO_DIR
        / "temporary_visualization_batches"
    )
    temporary_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    batches = list(
        split_into_batches(
            test_entries,
            BATCH_SIZE,
        )
    )

    print("=" * 80)
    print("BATCH RECONSTRUCTION")
    print("=" * 80)
    print(f"Pazienti totali:      {len(test_entries)}")
    print(f"Pazienti per gruppo:  {BATCH_SIZE}")
    print(f"Numero di gruppi:     {len(batches)}")
    print(f"Superficie:           {SURFACE}")
    print(f"Versione:             {version_name}")
    print(f"Campioni per paziente:{NUM_SAMPLES}")
    print("=" * 80)

    for batch_index, batch_entries in enumerate(batches):
        patient_names = [
            get_patient_name(entry)
            for entry in batch_entries
        ]

        print("\nPazienti del gruppo:")
        for patient in patient_names:
            print(f"  - {patient}")

        if DELETE_OLD_RECONSTRUCTIONS:
            delete_old_meshes(
                reconstructed_meshes_dir=reconstructed_meshes_dir,
                patient_names=patient_names,
                surface=SURFACE,
            )

        batch_json = reconstruct_batch(
            batch_entries=batch_entries,
            batch_index=batch_index,
            temporary_dir=temporary_dir,
            reconstructed_meshes_dir=reconstructed_meshes_dir,
        )

        plot_batch(
            patient_names=patient_names,
            reconstructed_meshes_dir=reconstructed_meshes_dir,
            version_name=version_name,
            batch_index=batch_index,
            total_batches=len(batches),
        )

        delete_old_meshes(
            reconstructed_meshes_dir=reconstructed_meshes_dir,
            patient_names=patient_names,
            surface=SURFACE,
        )
        

        # Il JSON non serve più dopo il completamento del gruppo
        if batch_json.exists():
            batch_json.unlink()

    print("\nTutti i pazienti sono stati ricostruiti e visualizzati.")


if __name__ == "__main__":
    main()