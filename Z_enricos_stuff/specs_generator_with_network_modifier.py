#!/usr/bin/env python3

"""
This script generates specs_files/specs.json inside each architecture folder.

Each folder name must have the format:

    5k_D3_W64_L16

where:
    D = depth, e.g. 3 or 5
    W = width, e.g. 64 or 128
    L = latent size, e.g. 16 or 32

For each folder, the script updates:
    - TrainSplit
    - TestSplit
    - DataSource
    - NetworkSpecs["dims"]
    - NetworkSpecs["latent_size"]
"""

# region HowToRun
# python specs_generator_with_network_modifier.py 
#       -i root/folder/with/architectures/folders \ ex: /home/rizzardi/Schreibtisch/combinations/5k_architecture_exploration
#       -t path/to/templpate/specs.json \ ex: /home/rizzardi/workspace/deepcsdf-ventricles/specs_files/specs_deepsdfatria.json
#       --data-root /home/rizzardi/Schreibtisch/sampling_noise_study_npy/5k/S_0.025-L_0.75-R_0.5


from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from copy import deepcopy


# -------------------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Populate specs_files/specs.json inside each architecture folder"
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        type=Path,
        help="Root folder containing architecture subfolders"
    )

    parser.add_argument(
        "-t", "--template",
        required=True,
        type=Path,
        help="Path to template specs JSON file"
    )

    parser.add_argument(
        "--specs-name",
        default="specs.json",
        help="Output specs filename inside specs_files/<name>"
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Optional root folder containing the data folders. If not provided, it is derived automatically."
    )

    return parser.parse_args()


# -------------------------------------------------------------------------------------------
def load_json(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------------------------------------------------------------------------
def write_json(data: dict, json_path: Path):
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# -------------------------------------------------------------------------------------------
def parse_architecture_from_folder_name(folder_name: str):
    """
    Example folder name:
        5k_D3_W64_L16

    Returns:
        dims = [64, 64, 64]
        latent_size = 16
    """

    pattern = r".*_D(?P<depth>\d+)_W(?P<width>\d+)_L(?P<latent>\d+)$"
    match = re.match(pattern, folder_name)

    if match is None:
        raise ValueError(
            f"Folder name does not match expected format '*_D<depth>_W<width>_L<latent>': {folder_name}"
        )

    depth = int(match.group("depth"))
    width = int(match.group("width"))
    latent_size = int(match.group("latent"))

    if depth not in [3, 5, 7]:
        raise ValueError(
            f"Invalid depth {depth} in folder {folder_name}. Allowed values: 3, 5, 7"
        )

    if width not in [64, 128, 256, 512]:
        raise ValueError(
            f"Invalid width {width} in folder {folder_name}. Allowed values: 64, 128, 256"
        )

    if latent_size not in [16, 32, 64, 128]:
        raise ValueError(
            f"Invalid latent size {latent_size} in folder {folder_name}. Allowed values: 16, 32, 64"
        )

    dims = [width] * depth

    return dims, latent_size


# -------------------------------------------------------------------------------------------
def build_specs_for_combination(
    template_specs: dict,
    combo_dir: Path,
    data_root: Path,
) -> dict:

    specs = deepcopy(template_specs)

    dims, latent_size = parse_architecture_from_folder_name(combo_dir.name)

    specs["TrainSplit"] = str((combo_dir / "train" / "data_fnames_train.json").resolve())
    specs["TestSplit"] = str((combo_dir / "test" / "data_fnames_test.json").resolve())

    # 👇 ORA È FISSO
    specs["DataSource"] = str(data_root.resolve())

    specs["Network_specs"]["dims"] = dims
    specs["Network_specs"]["latent_size"] = latent_size

    return specs


# -------------------------------------------------------------------------------------------
def main():
    args = parse_args()

    combinations_root = args.input
    template_path = args.template
    specs_name = args.specs_name

    if not combinations_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {combinations_root}")

    if not template_path.exists():
        raise FileNotFoundError(f"Template file does not exist: {template_path}")

    if args.data_root is not None:
        data_root = args.data_root
    else:
        data_root = Path("/home/rizzardi/Schreibtisch/sampling_noise_study_npy") / combinations_root.name

    if not data_root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_root}")

    template_specs = load_json(template_path)

    combo_dirs = sorted([
        p for p in combinations_root.iterdir()
        if p.is_dir()
    ])

    print(f"[INFO] Found {len(combo_dirs)} architecture folders.")

    for idx, combo_dir in enumerate(combo_dirs, start=1):
        combo_name = combo_dir.name

        print(f"[{idx}/{len(combo_dirs)}] Processing: {combo_name}")

        if not data_root.exists():
            raise FileNotFoundError(f"Data directory does not exist: {data_root}")
        
        specs_dir = combo_dir / "specs_files"
        specs_dir.mkdir(parents=True, exist_ok=True)

        specs = build_specs_for_combination(
            template_specs=template_specs,
            combo_dir=combo_dir,
            data_root=data_root,
        )

        out_path = specs_dir / specs_name
        write_json(specs, out_path)

        print(f"[{idx}/{len(combo_dirs)}] Created: {out_path}")
        print(f"          dims = {specs['Network_specs']['dims']}")
        print(f"          latent_size = {specs['Network_specs']['latent_size']}")

    print("[INFO] Done.")


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()