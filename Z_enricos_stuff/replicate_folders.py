"""
this code simply creates in a directory folders of the same name of those present in anoter directory.
"""

# region HowToRun
# pyhton replicate_folders.py -i /path/input_dir -o path/output_dir
# endregion

import argparse
from pathlib import Path


#------------------------------------------------------------------------------------------------------------
def replicate_folders(input_dir: Path, output_dir: Path):
    if not input_dir.exists():
        raise FileNotFoundError(f"input directory does not exists: {input_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)

    folders = [p for p in input_dir.iterdir() if p.is_dir()]

    print("[info] original folders are 27.")
    print(f"[info] found {len(folders)} folders in input.")
    print(f"found {len(folders)}/27 folders.")

    for folder in folders:
        new_folder = output_dir / folder.name
        new_folder.mkdir(parents=True, exist_ok=True)
        print(f"[OK] created: {new_folder}")


#------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="replicate folder names frominput dir to output dir")
    parser.add_argument("--input",
                        required=True,
                        help="path to input directory")
    
    parser.add_argument("--output",
                        required=True,
                        help="path to output directory")
    
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    replicate_folders(input_dir, output_dir)


#------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()