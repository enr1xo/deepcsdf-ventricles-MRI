"""
This file generates the specs_files for each combination, they will all be equal except for 
    - "TrainSplit"
    - "TestSplit"
    - "DataSource"
that will point to the split of the i-th combination
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from copy import deepcopy

#-------------------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Populate specs_files/specs.json inside each combination folder"
    )
    parser.add_argument(
        "-i", "--input", required=True, type=Path,
        help="Root folder containing combination subfolders"
    )
    parser.add_argument(
        "-t", "--template", required=True, type=Path,
        help="Path to template specs JSON file"
    )
    parser.add_argument(
        "--specs-name", default="specs.json",
        help="output specs filename inside specs_files / <name>"
    )
    return parser.parse_args()

#-------------------------------------------------------------------------------------------
def load_json(json_path:Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)

#-------------------------------------------------------------------------------------------
def write_json(data:dict, json_path : Path):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

#---------------------------------------------------------------------------------------
def build_specs_for_combination(template_specs: dict, combo_dir: Path, data_combo_dir: Path) -> dict:
    specs = deepcopy(template_specs)

    specs["TrainSplit"] = str((combo_dir / "train" / "data_fnames_train.json").resolve())
    specs["TestSplit"] = str((combo_dir / "test" / "data_fnames_test.json").resolve())
    specs["DataSource"] = str(data_combo_dir.resolve())

    return specs

#---------------------------------------------------------------------------------------
def main():
    args = parse_args()

    combinations_root = args.input
    template_path = args.template
    specs_name = args.specs_name

    if not combinations_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {combinations_root}")

    if not template_path.exists():
        raise FileNotFoundError(f"Template file does not exist: {template_path}")

    # ricava automaticamente la root dati
    data_root = Path("/home/rizzardi/Schreibtisch/sampling_noise_study_npy") / combinations_root.name

    if not data_root.exists():
        raise FileNotFoundError(f"Derived data directory does not exist: {data_root}")

    template_specs = load_json(template_path)

    combo_dirs = sorted([
        p for p in combinations_root.iterdir()
        if p.is_dir() and p.name.startswith("S_")
    ])

    print(f"[INFO] Found {len(combo_dirs)} combination folders.")

    for idx, combo_dir in enumerate(combo_dirs, start=1):
        combo_name = combo_dir.name
        data_combo_dir = data_root / combo_name

        print(f"[{idx}/{len(combo_dirs)}] Processing: {combo_name}")

        if not data_combo_dir.exists():
            print(f"[WARNING] Missing data folder: {data_combo_dir}")
            continue

        specs_dir = combo_dir / "specs_files"
        specs_dir.mkdir(parents=True, exist_ok=True)

        specs = build_specs_for_combination(
            template_specs=template_specs,
            combo_dir=combo_dir,
            data_combo_dir=data_combo_dir,
        )

        out_path = specs_dir / specs_name
        write_json(specs, out_path)

        print(f"[{idx}/{len(combo_dirs)}] Created: {out_path}")

    print("[INFO] Done.")
#-------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()