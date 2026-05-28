#!/usr/bin/env python3

import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import time
import argparse


# =========================
# CONFIG
# =========================
# COMBINATIONS_ROOT = Path("/home/rizzardi/Schreibtisch/combinations/5k_architecture_exploration/combs_data")
COMBINATIONS_ROOT = Path("/home/rizzardi/Schreibtisch/combinations/5k_architecture_exploration_extended/combs_data")
TEST_ONE_SCRIPT = Path("/home/rizzardi/workspace/deepcsdf-ventricles/Z_enricos_stuff/test_one_for_parallel.py")

MAX_PARALLEL_TESTS = 2


# =========================
# ARGPARSE
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Run test in parallel on all combinations")
    parser.add_argument(
        "--metric",
        required=True,
        choices=["chamfer", "haussdorff", "lddmm"],
        help="Metric to compute (ONLY ONE per run!)"
    )
    parser.add_argument(
        "--experiment_name",
        default="architecture_exploration",
        help="Experiment name"
    )
    parser.add_argument(
        "--version",
        default="latest",
        help="Version to test"
    )
    parser.add_argument(
        "--dataset_split",
        default="test",
        choices=["train", "test"],
        help="which split to evaluate"
    )
    return parser.parse_args()


# =========================
# HELPERS
# =========================
def is_valid_combination_dir(combo_dir: Path) -> bool:
    return (
        combo_dir.is_dir()
        and (combo_dir / "specs_files" / "specs.json").exists()
        and (combo_dir / "train").exists()
        and (combo_dir / "test").exists()
    )


# =========================
# RUN ONE TEST
# =========================
def run_one_test(combo_dir: Path, metric: str, experiment_name: str, version: str, dataset_split: str):
    combo_name = combo_dir.name

    override_dataset = f"{dataset_split}/data_fnames_{dataset_split}.json"

    cmd = [
        "python",
        str(TEST_ONE_SCRIPT),
        "--combo_dir", str(combo_dir),
        "--experiment_name", experiment_name,
        "--version", version,
        "--mode", "1",
        "--override_with_dataset", override_dataset,
    ]

    if metric == "chamfer":
        cmd.append("--compute_chamfer")
    elif metric == "haussdorff":
        cmd.append("--compute_haussdorff")
    elif metric == "lddmm":
        cmd.append("--compute_lddmm")

    print(f"[START] {combo_name} ({metric})")
    tic = time()

    try:
        completed = subprocess.run(
            cmd,
            text=True,
            check=True
        )

        elapsed = time() - tic

        return {
            "combo": combo_name,
            "success": True,
            "elapsed_sec": elapsed,
        }

    except subprocess.CalledProcessError as e:
        elapsed = time() - tic

        return {
            "combo": combo_name,
            "success": False,
            "elapsed_sec": elapsed,
        }


# =========================
# MAIN
# =========================
def main():
    args = parse_args()

    metric = args.metric
    experiment_name = args.experiment_name
    version = args.version
    dataset_split = args.dataset_split

    combo_dirs = sorted([
        p for p in COMBINATIONS_ROOT.iterdir()
        if is_valid_combination_dir(p)
    ])

    print(f"[INFO] Found {len(combo_dirs)} valid combinations.")
    print(f"[INFO] Running metric: {metric}")

    if not combo_dirs:
        print("[INFO] No valid combination folders found.")
        return

    results = []

    # === SEQUENZIALE ===
    if MAX_PARALLEL_TESTS == 1:
        for idx, combo_dir in enumerate(combo_dirs, start=1):
            print(f"\n[INFO] Test {idx}/{len(combo_dirs)}: {combo_dir.name}")

            result = run_one_test(combo_dir, metric, experiment_name, version, dataset_split)
            results.append(result)

            status = "OK" if result["success"] else "ERROR"
            print(f"[{status}] {result['combo']} - {result['elapsed_sec'] / 60:.2f} min")

    # === PARALLELO ===
    else:
        with ProcessPoolExecutor(max_workers=MAX_PARALLEL_TESTS) as executor:
            futures = {
                executor.submit(run_one_test, combo_dir, metric, experiment_name, version, dataset_split): combo_dir.name
                for combo_dir in combo_dirs
            }

            for idx, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)

                status = "OK" if result["success"] else "ERROR"
                print(
                    f"[{idx}/{len(combo_dirs)}] [{status}] {result['combo']} "
                    f"- {result['elapsed_sec'] / 60:.2f} min"
                )

    # =========================
    # SUMMARY
    # =========================
    n_ok = sum(1 for r in results if r["success"])
    n_fail = len(results) - n_ok

    print("\n[SUMMARY]")
    print(f"Metric: {metric}")
    print(f"Success: {n_ok}")
    print(f"Failed: {n_fail}")


if __name__ == "__main__":
    main()