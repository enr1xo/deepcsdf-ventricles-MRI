#!/usr/bin/env python3

from __future__ import annotations

import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import time


# COMBINATIONS_ROOT = Path("/home/rizzardi/Schreibtisch/combinations/5k_architecture_exploration/combs_data")
COMBINATIONS_ROOT = Path("/home/rizzardi/Schreibtisch/combinations/5k_architecture_exploration_extended/combs_data")
TRAIN_ONE_SCRIPT = Path("/home/rizzardi/workspace/deepcsdf-ventricles/Z_enricos_stuff/train_one_for_parallel.py")

MAX_PARALLEL_TRAININGS = 3
DATALOADER_WORKERS = 8
SHOW_PROGRESS = True

folder_initial_characters = "5k_"

# -----------------------------------------------------------------------------
def is_valid_combination_dir(combo_dir: Path) -> bool:
    return (
        combo_dir.is_dir()
        and combo_dir.name.startswith(folder_initial_characters)
        and (combo_dir / "config.py").exists()
        and (combo_dir / "specs_files" / "specs.json").exists()
        and (combo_dir / "train" / "data_fnames_train.json").exists()
        and (combo_dir / "test" / "data_fnames_test.json").exists()
        and (combo_dir / "experiments").exists()
    )


# -----------------------------------------------------------------------------
def run_one_training(combo_dir: Path) -> dict:
    combo_dir = Path(combo_dir)
    combo_name = combo_dir.name

    cmd = [
        "python",
        str(TRAIN_ONE_SCRIPT),
        "--combo_dir", str(combo_dir),
        "--experiment_name", "architecture_exploration",
        "--num_workers_dataloader", str(DATALOADER_WORKERS),
        "--show_progress",
    ]

    if SHOW_PROGRESS:
        cmd.append("--show_progress")

    print(f"[START] {combo_name}")
    tic = time()

    try:
        completed = subprocess.run(
            cmd,
            # capture_output=False,
            text=True,
            check=True,
        )
        elapsed = time() - tic

        return {
            "combo": combo_name,
            "combo_dir": combo_dir,
            "success": True,
            "elapsed_sec": elapsed,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    except subprocess.CalledProcessError as e:
        elapsed = time() - tic
        return {
            "combo": combo_name,
            "combo_dir": combo_dir,
            "success": False,
            "elapsed_sec": elapsed,
            "stdout": e.stdout,
            "stderr": e.stderr,
        }


# -----------------------------------------------------------------------------
def write_launcher_log(result: dict):
    combo_dir = Path(result["combo_dir"])
    log_path = combo_dir / "launcher_log.txt"

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== STDOUT ===\n")
        f.write(result["stdout"] or "")
        f.write("\n\n=== STDERR ===\n")
        f.write(result["stderr"] or "")


# -----------------------------------------------------------------------------
def main():
    if not COMBINATIONS_ROOT.exists():
        raise FileNotFoundError(f"Combinations root does not exist: {COMBINATIONS_ROOT}")

    if not TRAIN_ONE_SCRIPT.exists():
        raise FileNotFoundError(f"Training script does not exist: {TRAIN_ONE_SCRIPT}")

    combo_dirs = sorted([
        p for p in COMBINATIONS_ROOT.iterdir()
        if is_valid_combination_dir(p)
    ])

    print(f"[INFO] Found {len(combo_dirs)} valid combinations.")

    if not combo_dirs:
        print("[INFO] No valid combination folders found.")
        return

    results = []

    if MAX_PARALLEL_TRAININGS == 1:
        for idx, combo_dir in enumerate(combo_dirs, start=1):
            print(f"\n[INFO] Training {idx}/{len(combo_dirs)}: {combo_dir.name}")
            result = run_one_training(combo_dir)
            results.append(result)

            status = "OK" if result["success"] else "ERROR"
            print(
                f"[{status}] {result['combo']} "
                f"- {result['elapsed_sec'] / 60:.2f} min"
            )

            write_launcher_log(result)

    else:
        with ProcessPoolExecutor(max_workers=MAX_PARALLEL_TRAININGS) as executor:
            futures = {
                executor.submit(run_one_training, combo_dir): combo_dir.name
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

                write_launcher_log(result)

    n_ok = sum(1 for r in results if r["success"])
    n_fail = len(results) - n_ok

    print("\n[SUMMARY]")
    print(f"Success: {n_ok}")
    print(f"Failed: {n_fail}")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()