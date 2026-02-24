#!/usr/bin/env python3
r"""
Extract unique tag values from VTK files in a directory.

Usage:
C:\Users\e.rizzardi\OneDrive\Desktop\grazproject\deepcsdf-ventricles\extract_tags_from_vtk.py
  python extract_tags_from_vtk.py /path/to/vtk_dir --field tag
"""

import argparse
from pathlib import Path
import vtk


def extract_unique_values(vtk_file: Path, field_name: str, location: str):
    """Return sorted unique values from a VTK file."""
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(str(vtk_file))
    reader.Update()

    grid = reader.GetOutput()

    if location == "cell":
        data = grid.GetCellData()
    else:
        data = grid.GetPointData()

    array = data.GetArray(field_name)
    if array is None:
        raise ValueError(f"Field '{field_name}' not found in {location} data")

    values = set()
    for i in range(array.GetNumberOfTuples()):
        values.add(array.GetValue(i))

    return sorted(values)


def main():
    parser = argparse.ArgumentParser(description="Extract unique tag values from VTK files.")
    parser.add_argument("directory", type=Path, help="Directory containing .vtk files")
    parser.add_argument("--field", required=True, help="Name of the tag field (e.g. tag, RegionId)")
    parser.add_argument("--location", choices=["cell", "point"], default="cell",
                        help="Where the field is stored (default: cell)")
    parser.add_argument("--save", action="store_true",
                        help="Save tags to a .txt file next to each VTK")
    args = parser.parse_args()

    vtk_files = sorted(args.directory.glob("*.vtk"))
    if not vtk_files:
        print(f"No .vtk files found in {args.directory}")
        return

    for vtk_file in vtk_files:
        try:
            tags = extract_unique_values(vtk_file, args.field, args.location)
            print(f"\n{vtk_file.name}")
            print(f"  {args.field} values: {tags}")

            if args.save:
                out = vtk_file.with_suffix(f".{args.field}_values.txt")
                with open(out, "w") as f:
                    for t in tags:
                        f.write(f"{t}\n")

        except Exception as e:
            print(f"\n{vtk_file.name}")
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
