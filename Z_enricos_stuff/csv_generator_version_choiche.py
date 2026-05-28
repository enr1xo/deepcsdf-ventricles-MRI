#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import pandas as pd


# --------------------------------------------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("yes", "true", "t", "1")


# --------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Scorre tutte le cartelle combinazione dentro un root, "
            "legge i CSV in <combinazione>/results/metrics/architecture_exploration, "
            "calcola media e std per ogni superficie e metrica, "
            "salva un CSV riassuntivo."
        )
    )

    parser.add_argument(
        "-i", "--input_root",
        required=True,
        help="root che contiene le cartelle combinazione"
    )

    parser.add_argument(
        "-o", "--output_csv",
        required=True,
        help="cartella dove salvare il csv finale"
    )

    parser.add_argument(
        "-n", "--name_csv",
        choices=["3k", "5k"],
        required=True,
        help="nome del dataset/combinazione, es. 3k o 5k"
    )

    parser.add_argument(
        "--analyze_lddmm",
        type=str2bool,
        required=True,
        help="true/false: se analizzare anche la LDDMM"
    )

    parser.add_argument(
        "-v", "--version",
        default="latest",
        help="versione da analizzare, es. version_0, version_1 oppure latest"
    )

    return parser.parse_args()


# --------------------------------------------------------------
def get_latest_metric_version(metrics_dir: Path, experiment_name: str) -> str:
    """
    Cerca nei nomi dei file CSV la versione più alta.

    Esempio file:
        architecture_exploration-version_0-chamfer-test.csv
        architecture_exploration-version_1-haussdorff-test.csv

    Ritorna:
        version_1
    """
    pattern = re.compile(
        rf"^{re.escape(experiment_name)}-version_(\d+)-.*\.csv$"
    )

    versions = []

    for csv_file in metrics_dir.glob("*.csv"):
        match = pattern.match(csv_file.name)
        if match:
            versions.append(int(match.group(1)))

    if not versions:
        raise FileNotFoundError(
            f"No metric CSVs found with pattern "
            f"{experiment_name}-version_*-*.csv in {metrics_dir}"
        )

    return f"version_{max(versions)}"


# --------------------------------------------------------------
def find_metric_name(df: pd.DataFrame, csv_path: Path) -> str:
    """
    Ricava il nome della metrica dalla colonna 'metric'.
    Se non è possibile, lo ricava dal nome del file.
    """
    if "metric" in df.columns:
        metric_values = df["metric"].dropna().astype(str).unique()
        if len(metric_values) == 1:
            return metric_values[0]

    # fallback dal nome file:
    # architecture_exploration-version_0-chamfer-test.csv -> chamfer
    parts = csv_path.stem.split("-")
    if len(parts) >= 4:
        return parts[2]

    return csv_path.stem


# --------------------------------------------------------------
def summarize_metric_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=None, engine="python")

    required_cols = {"organ", "value"}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(f"missing columns {missing_cols}")

    metric_name = find_metric_name(df, csv_path)

    summary = (
        df.groupby("organ")["value"]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "organ",
        f"{metric_name}_mean",
        f"{metric_name}_std",
    ]

    return summary


# --------------------------------------------------------------
def process_combination(
    combo_dir: Path,
    analyze_lddmm: bool,
    version: str = "latest",
    experiment_name: str = "architecture_exploration",
) -> pd.DataFrame | None:

    metrics_dir = combo_dir / "results" / "metrics" / experiment_name

    if not metrics_dir.exists():
        print(f"WARNING - missing: {metrics_dir}")
        return None

    selected_version = version

    if selected_version == "latest":
        try:
            selected_version = get_latest_metric_version(
                metrics_dir,
                experiment_name
            )
        except Exception as e:
            print(f"WARNING - cannot find latest version in {metrics_dir}: {e}")
            return None

    csv_files = sorted(
        metrics_dir.glob(f"{experiment_name}-{selected_version}-*.csv")
    )

    if not csv_files:
        print(
            f"WARNING - no CSV files for "
            f"{experiment_name}-{selected_version} in {metrics_dir}"
        )
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


# --------------------------------------------------------------
def main():
    args = parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_csv)

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = (
        output_dir /
        f"metrics_csv_and_std_27combs_{args.name_csv}_{args.version}.csv"
    )

    results = []

    for comb_dir in sorted([p for p in input_root.iterdir() if p.is_dir()]):
        print(f"INFO - processing {comb_dir.name}")

        res = process_combination(
            comb_dir,
            args.analyze_lddmm,
            args.version,
        )

        if res is not None:
            results.append(res)

    if not results:
        raise ValueError("nessun risultato valido.")

    final_df = pd.concat(results, ignore_index=True)

    cols_base = ["combination", "version", "organ"]
    cols_other = [c for c in final_df.columns if c not in cols_base]
    final_df = final_df[cols_base + cols_other]

    final_df.to_csv(output_file, index=False)

    print(f"INFO - csv salvato in {output_file}")


# --------------------------------------------------------------
if __name__ == "__main__":
    main()