"""
This code remove the characters '[' and ']' from a csv file.
"""

# region HowToRun
# python remove_square_brakets.py -input path/to/folder/where/csv/file/is
# endregion

"""
How does it work:
    It look for all the subfolders in '-input' looking for a csv file,
    one this is file is found, it removes the square brakets from the column 'value'
"""

import argparse
import pandas as pd
from pathlib import Path

#------------------------------------------------------
def process_csv(csv_path):

    try:
        df = pd.read_csv(csv_path)

        if "value" not in df.columns:
            print(f"skipping, 'value' non in found in column of {csv_path}")
            return
        
        df["value"] = df["value"].astype(str).str.replace(r"[\[\]]", "", regex=True)

        df.to_csv(csv_path, index=False)

        print(f"ok, processed {csv_path}")
    
    except Exception as e:
        print(f"error: {csv_path}: {e}")


#----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="remove square brackets fom csv file.")
    parser.add_argument(
        "-input",
        required=True,
        help="Path to root folder of the experiment metric. ex: /home/rizzardi/Schreibtisch/5fold_test_results/lddmm"
    )

    args = parser.parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"error: {input_path} does not exists")

        return
    
    csv_files = list(input_path.rglob("*.csv"))

    if not csv_files:
        print(f"error: no csv files found")
    
    if len(csv_files) != 20:
        print(f"found {len(csv_files)} / 20")
    
    for csv_file in csv_files:
        process_csv(csv_file)


#-------------------------------------------------------------
if __name__ == "__main__":
    main()