"""
Questo codice scorre in tutte le cartelle del path passato in -i.
Ognuna delle quali rappresenta una combinazione di parametri nella funzione di sampling. 
In ogni cartelle di ogni combinazione in 
    <combinazione>/results/metrics/noise_study
troverà 3 file csv, ognuno relativo a una metrica calcolata sulle predizioni delle tre superfici dei pazienti di test.
Per ogni CSV calcola media e deviazione standard delle misure per ciascuna superficie e salva un CSV finale in -o, 
    con una riga per superficie per ogni combinazione e, per ciascuna metrica, le colonne di media e deviazione standard.
"""

# region HowToRun
# python combs_study_csv_generator.py \
#       -i root del folder che contiene i folder delle combinazioni \
#       -o root dove salvare il csv
#       -n name of the csv file (just put 3k or 5k)
#       --analyze_lddmm true/false
# endregion

import argparse
from pathlib import Path

import pandas as pd
import re

#-----------------------------------------------------------
def get_latest_version_dir2(metrics_root: Path) -> Path:
    version_dirs = [
        p for p in metrics_root.iterdir()
        if p.is_dir() and p.name.startswith("version_")
    ]

    if not version_dirs:
        raise FileNotFoundError(f"No version_* folders found in {metrics_root}")

    return sorted(
        version_dirs,
        key=lambda p: int(p.name.split("_")[-1])
    )[-1]

def get_latest_version_dir(experiments_root: Path) -> Path:
    version_dirs = [
        p for p in experiments_root.rglob("version_*")
        if p.is_dir() and re.fullmatch(r"version_\d+", p.name)
    ]

    if not version_dirs:
        raise FileNotFoundError(f"No version_* folders found in {experiments_root}")

    return max(
        version_dirs,
        key=lambda p: int(p.name.split("_")[-1])
    )

#--------------------------------------------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("yes", "true", "t", "1")
#------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=("scorre tutte le cartelle combinazione dentro un root," \
        "legge i CSV in <combinazione>/results/metrics/noise_study, calcola media e std per ogni superifice e metrics," \
        "salva un csv riassuntivo.")
    )
    parser.add_argument(
        "-i", "--input_root",
        required=True,
        help="root che contiene le cartelle combinazione"
    )
    parser.add_argument(
        "-o", "--output_csv",
        required=True,
        help="path completo di dove salvare il csv"
    )
    parser.add_argument(
        "-n", "--name_csv",
        choices=["3k", "5k"],
        required=True,
        help="nome di come salvare il csv"
    )
    parser.add_argument(
        "--analyze_lddmm",
        type=str2bool,
        required=True,
        help="booleano per deciedere se analizzare o meno anche la lddmm"
    )
    parser.add_argument(
        "-v", "--version",
        default="latest",
        help="versione da utilizzare, es version_0, version_5 oppure"
    )

    return parser.parse_args()

#-------------------------------------------------------------------------------
def find_metric_name(df: pd.DataFrame, csv_path: Path) -> str:
    """
    ricava il nome della metrica dalla colonna metric del file, altrimenti usa il nome del file.
    """
    if "metric" in df.columns:
        metric_values = df["metric"].dropna().astype(str).unique()
        if len(metric_values) == 1:
            return metric_values[0]
    
    return csv_path.stem


#----------------------------------------------------------
def summarize_metric_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=None, engine="python")

    metric_name = find_metric_name(df, csv_path)

    summary = (
        df.groupby("organ")["value"]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = ["organ", f"{metric_name}_mean", f"{metric_name}_std"]

    return summary

#---------------------------------------------------------------------
def process_combination(combo_dir: Path, analyze_lddmm: bool, version: str = "latest") -> pd.DataFrame | None:
    experiments_root = combo_dir / "experiments" / "architecture_exploration"

    if not experiments_root.exists():
        print(f"WARNING - missing: {experiments_root}")
        return None

    if version == "latest":
        try:
            version_dir = get_latest_version_dir(experiments_root)
        except FileNotFoundError as e:
            print(f"WARNING - {e}")
            return None
    else:
        version_dir = experiments_root / version
        if not version_dir.exists():
            print(f"WARNING - missing version: {version_dir}")
            return None

    selected_version = version_dir.name
    print(f"INFO - using {selected_version} for {combo_dir.name}")

    metrics_dir = combo_dir / "results" / "metrics" / "architecture_exploration"

    if not metrics_dir.exists():
        print(f"WARNING - missing: {metrics_dir}")
        return None

    csv_files = sorted(metrics_dir.glob(f"*-{selected_version}-*.csv"))

    if not csv_files:
        print(f"WARNING - no CSV files found for {selected_version} in {metrics_dir}")
        return None

    merged = None

    for csv_file in csv_files:
        if not analyze_lddmm and "lddmm" in csv_file.name.lower():
            continue

        try:
            summary = summarize_metric_csv(csv_file)
        except Exception as e:
            print(f"WARNING - error {csv_file}: {e}")
            continue

        if merged is None:
            merged = summary
        else:
            merged = merged.merge(summary, on="organ", how="outer")

    if merged is None:
        return None

    merged.insert(0, "version", selected_version)
    merged.insert(0, "combination", combo_dir.name)

    return merged

#-----------------------------------------------------
def main():
    args = parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_csv)

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"metrics_csv_and_std_Cs_param_study_{args.name_csv}.csv"

    results = []

    for comb_dir in sorted([p for p in input_root.iterdir() if p.is_dir()]):
        print(f"INFO - processing {comb_dir.name}")

        res = process_combination(comb_dir, args.analyze_lddmm, args.version)
        if res is not None:
            results.append(res)

    if not results:
        raise ValueError("nessun risultato valido.")
    
    final_df = pd.concat(results, ignore_index=True)

    cols_base = ["combination", "organ"]
    cols_other = [c for c in final_df.columns if c not in cols_base]
    final_df = final_df[cols_base + cols_other]

    final_df.to_csv(output_file, index=False)

    print(f"INFO - csv salvato in {output_file}")


#---------------------------------------
if __name__ == "__main__":
    main()