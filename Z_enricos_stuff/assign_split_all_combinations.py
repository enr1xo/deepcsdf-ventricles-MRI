"""
This code creates for each subfolder in a directory two subfoders (test and train), and each of them creates
    a .json file of the names of the patients to use for train or test.
"""

# region HowToRun
# python assign_split_all_combinations.py -i path/container_folder -k path/folder_of_split_files
# endregion

from __future__ import annotations

import argparse
import json

from pathlib import Path


#---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Assign train/test JSON splits inside each combination folder"
    )
    parser.add_argument(
        "--data_root", required=True, type=Path,
        help="Root folder containing the .npy combination folders"
    )
    parser.add_argument(
        "--output_root", required=True, type=Path,
        help="Root folder containing the combination folders where train/test JSONs will be written"
    )
    parser.add_argument(
        "-k", "--keys", required=True, type=Path,
        help="Folder containing train.txt and test.txt"
    )
    parser.add_argument(
        "--pattern", default="*.npy", type=str,
        help="File pattern to search inside each combination folder (default: *.npy)"
    )
    return parser.parse_args()


#---------------------------------------------------------------------------
def extract_key(filename: str) -> str:
    if "-epi_lv_rv_" in filename:
        return filename.split("-epi_lv_rv_")[0]

    if "_MRI_like_" in filename:
        return filename.split("_MRI_like_")[0]

    raise ValueError(f"Unexpected filename format: {filename}")

#---------------------------------------------------------------------------
def load_keys(filepath: Path) -> set[str]:
    with filepath.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}

#---------------------------------------------------------------------------
def assign_split_for_one_combination(
    input_dir: Path,
    keys_dir: Path,
    output_dir: Path,
    pattern: str = "*.npy",
):
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"

    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    train_keys = load_keys(keys_dir / "train.txt")
    test_keys = load_keys(keys_dir / "test.txt")

    train_files = []
    test_files = []
    unknown_keys = set()

    files = sorted(input_dir.rglob(pattern))

    for f in files:
        key = extract_key(f.name)

        if key in train_keys:
            train_files.append(f.name)
        elif key in test_keys:
            test_files.append(f.name)
        else:
            unknown_keys.add(key)

    with (train_dir / "data_fnames_train.json").open("w", encoding="utf-8") as f:
        json.dump(train_files, f, indent=2)

    with (test_dir / "data_fnames_test.json").open("w", encoding="utf-8") as f:
        json.dump(test_files, f, indent=2)

    return {
        "total_files": len(files),
        "train_files": len(train_files),
        "test_files": len(test_files),
        "unknown_keys": sorted(unknown_keys),
    }

#---------------------------------------------------------------------------
def main():
    args = parse_args()

    data_root = args.data_root
    output_root = args.output_root
    keys_dir = args.keys
    pattern = args.pattern

    if not data_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {data_root}")


    if not output_root.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_root}")
    
    if not keys_dir.exists():
        raise FileNotFoundError(f"Keys directory does not exist: {keys_dir}")

    if not (keys_dir / "train.txt").exists():
        raise FileNotFoundError(f"Missing file: {keys_dir / 'train.txt'}")

    if not (keys_dir / "test.txt").exists():
        raise FileNotFoundError(f"Missing file: {keys_dir / 'test.txt'}")

    output_combo_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])

    print(f"[INFO] Found {len(output_combo_dirs)} combination folders.")

    for idx, combo_dir in enumerate(output_combo_dirs, start=1):
        combo_name = combo_dir.name
        input_combo_dir = data_root / combo_name
        output_combo_dir = output_root #/ combo_name

        print(f"\n[{idx}/{len(output_combo_dirs)}] Processing: {combo_name}")

        if not input_combo_dir.exists():
            print(f"  [WARNING] Missing input data folder: {input_combo_dir}")
            continue

        result = assign_split_for_one_combination(
            input_dir=input_combo_dir,
            keys_dir=keys_dir,
            output_dir=output_combo_dir,
            pattern=pattern,
        )

        print(f"  Total files: {result['total_files']}")
        print(f"  Train files: {result['train_files']}")
        print(f"  Test files:  {result['test_files']}")

        if result["unknown_keys"]:
            print(f"  WARNING: {len(result['unknown_keys'])} keys not found in split")
            for k in result["unknown_keys"][:10]:
                print(f"    {k}")
            if len(result["unknown_keys"]) > 10:
                print("    ...")

#---------------------------------------------------------------------------
if __name__ == "__main__":
    main()