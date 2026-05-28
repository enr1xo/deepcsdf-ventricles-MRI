"""
This code generates a config.py file for each combination-folder with the corret paths pointing at the test and train files, the .npy files ecc
"""

# region HowToRun
# python configs_genrator.py -i path/to_combinations_folder (/home/rizzardi/Schreibtisch/combinations/3k)
# endregion

# region BE CAREFULL!!
# inside the function build_config, the term 'PATIENTS_COORDS_AND_SDFS_DIR' is hardcoded, 
# should be changed with the path where the .npy data of the combination set you are working with are!!!
# endregion

from __future__ import annotations

import argparse
from pathlib import Path

# per lo studio delle architetture dei 5k: --d /home/rizzardi/Schreibtisch/sampling_noise_study_npy/5k/S_0.025-L_0.75-R_0.5
#------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate config.py inside each combination folder"
    )
    parser.add_argument(
        "-i", "--input", required=True, type=Path,
        help="Root folder containing the combination subfolders"
    )
    parser.add_argument(
        "-d", "--data", required=True, type=Path,
        help="Root folder containing the .npy data subfolders for each combination"
    )
    return parser.parse_args()

#------------------------------------------------------------------------
def build_config(data_combo_dir: Path) -> str:
    return f'''from pathlib import Path
import os

# Paths
PROJ_ROOT = Path(__file__).resolve().parent
print(PROJ_ROOT)

if os.name == "nt":
    PATIENT_MESHES_DIR = Path(r"C:\\Users\\e.rizzardi\\OneDrive\\Desktop\\SDF_patients\\AF_patients")
else:
    PATIENT_MESHES_DIR = Path("/home/rizzardi/Schreibtisch/all_processed_files")

VENTRICLE_TAGS_METADATA = {{
    "CUSTOM_LABELS": {{
        "224": "RV_bridge", "223": "pulmunary_valve_rim", "222": "tricuspid_valve_rim",
        "205": "LV_bridge", "204": "aortic_valve:rim", "203": "Mitral_valve_rim",
        "8": "papillary_muscles"
    }},

    "LV_TAGS": [200, 201],
    "LV_ENDO_TAGS": [200],
    "LV_EPI_TAGS": [201],

    "RV_TAGS": [220, 221],
    "RV_ENDO_TAGS": [202, 220],
    "RV_EPI_TAGS": [221],

    "RA_TAGS": [99, 97, 95, 96, 94, 93, 92, 91, 90, 88, 87, 86, 85, 84, 48, 47, 46, 45, 44, 43, 42, 41],
    "RA_ENDO_TAGS": [41, 43, 45, 47, 86, 90, 91, 92, 93, 94, 97],
    "RA_EPI_TAGS": [42, 44, 46, 48, 84, 85, 86, 87, 88, 95, 99],

    "LA_TAGS": [82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 38, 37, 32, 31],
    "LA_ENDO_TAGS": [31, 37, 75, 76, 77, 78, 79, 82],
    "LA_EPI_TAGS": [32, 38, 70, 71, 72, 73, 74, 80, 82],

    "SHARED_TAGS": [89, 83],

    "vein_names": ["IPV", "SPV", "IVC", "SVC"],
    "valve_names": ["MV", "TV"]
}}

DATA_DIR = PROJ_ROOT
PATIENTS_COORDS_AND_SDFS_DIR = Path(r"{data_combo_dir}")
PATIENTS_NPY_DATA_DIR = PATIENTS_COORDS_AND_SDFS_DIR

SPECS_FILES_DIR = PROJ_ROOT / "specs_files"
EXPERIMENTS_DIR = PROJ_ROOT / "experiments"

TRAIN_DATA_DIR = DATA_DIR / "train"
TEST_DATA_DIR = DATA_DIR / "test"

MODELS_DIR = PROJ_ROOT / "models"

RESULTS_DIR = PROJ_ROOT / "results"
IMAGES_DIR = RESULTS_DIR / "images"
RECONSTRUCTED_MESHES_DIR = RESULTS_DIR / "reconstructed"
LATENTS_DIR = RESULTS_DIR / "fitted_latents"
METRICS_DIR = RESULTS_DIR / "metrics"
'''


#------------------------------------------------------------------------
def generate_config_for_combination(combo_dir: Path, data_combo_dir: Path):
    combo_dir.mkdir(parents=True, exist_ok=True)
    (combo_dir / "specs_files").mkdir(exist_ok=True)
    (combo_dir / "experiments").mkdir(exist_ok=True)
    (combo_dir / "results").mkdir(exist_ok=True)
    (combo_dir / "models").mkdir(exist_ok=True)

    config_path = combo_dir / "config.py"
    config_text = build_config(data_combo_dir)

    with config_path.open("w", encoding="utf-8") as f:
        f.write(config_text)
    
    print(f"[ok] created {config_path}")


#--------------------------------------------------------------------------
def main():
    args = parse_args()
    combination_root = args.input
    data_root = args.data

    if not combination_root.exists():
        raise FileNotFoundError(f"input directory does not exist: {combination_root}")
    
    if not data_root.exists():
        raise FileNotFoundError(f"data directory does not exist: {data_root}")

    # if you want to use for each combination its respective sampling distribution
    # combo_dirs = sorted([p for p in combination_root.iterdir() if p.is_dir() and p.name.startswith("S_")])

    # print(f"[info] found {len(combo_dirs)} / 8 combination folders.")

    # for idx, combo_dir in enumerate(combo_dirs, start=1):
    #     combo_name = combo_dir.name
    #     data_combo_dir = data_root / combo_name

    #     print(f"[{idx}/{len(combo_dirs)}] processing: {combo_dir.name}")

    #     if not data_combo_dir.exists():
    #         print(f"[warning] missing data folder: {data_combo_dir}")
    #         continue

    #     generate_config_for_combination(combo_dir, data_combo_dir)
    
    # if you want to use a common sampling distribution
    combo_dirs = sorted([p for p in combination_root.iterdir() if p.is_dir()])

    for idx, combo_dir in enumerate(combo_dirs, start=1):

        print(f"[{idx}/{len(combo_dirs)}] processing: {combo_dir.name}")

        generate_config_for_combination(combo_dir, data_root)

    print("DONE")



#--------------------------------------------------------------------------
if __name__ == "__main__":
    main()
