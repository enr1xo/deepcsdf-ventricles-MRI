import json
from pprint import pprint
from multiprocessing import Process, set_start_method
import sys
import os
from pathlib import Path
from train import train, check_override_specs_validity, deep_update
from loguru import logger
from contextlib import redirect_stdout, redirect_stderr


# ---------------------- Function to launch a single training ----------------------
def run_single_training(alpha, sigma, specs, log_dir, experiment_name, num_workers):
    logfile = log_dir / f"Lipalpha{alpha}_CodeReg{sigma}.log"
    with open(logfile, "w") as f, redirect_stdout(f), redirect_stderr(f):
        print(f"Starting training: alpha={alpha}, sigma={sigma}")
        train(specs, experiment_name=experiment_name, num_workers=num_workers)




if __name__ == "__main__":

    from config import SPECS_FILES_DIR, EXPERIMENTS_DIR

    NUM_WORKERS = 0
    # IMPORTANT: Set num_workers=0 in the DataLoader when launching multiple trainings
    # concurrently in the same Python session or via multiprocessing with 'spawn'.
    #
    # Why:
    # 1. Each training process spawned with 'spawn' re-imports the entire script.
    #    If num_workers > 0, each child process will attempt to start its own DataLoader worker
    #    subprocesses. Multiple child processes spawning workers simultaneously can lead to:
    #       - OS-level resource exhaustion (too many processes)
    #       - PyTorch multiprocessing conflicts
    #       - Errors like "DataLoader worker exited unexpectedly"
    #
    # 2. num_workers=0 runs data loading synchronously **inside the main training process**.
    #    This guarantees every batch is loaded correctly and avoids crashes, even with many
    #    concurrent training processes.
    #
    # 3. Trade-off:
    #    - Slightly slower batch loading because no parallel workers are used.
    #    - Much safer and stable when running 10–40 trainings concurrently on the same GPU.
    #
    # Summary: With num_workers=0, all data still loads correctly, just sequentially, avoiding
    # multiprocessing conflicts when launching multiple Lightning trainers in parallel.


    # # ---------------------- Setup GPU memory ----------------------
    # # optional: limit each process to a safe fraction of GPU memory
    # torch.cuda.set_per_process_memory_fraction(1/40, device=0)
    # torch.cuda.set_per_process_memory_growth(True)


    # ---------------------- Setup experiment ----------------------
    SPECS_BASE_FILE = "specs_deepsdfatria-temp.json"

    # If running inside Slurm, use $SLURM_JOB_ID, else fallback to PID or timestamp
    slurm_job_id = os.environ.get("SLURM_JOB_ID", None)
    if slurm_job_id is not None:
        LOG_DIR = EXPERIMENTS_DIR / f"logs-train-temp/{slurm_job_id}"
    else:
        import time
        LOG_DIR = EXPERIMENTS_DIR / f"logs-train-temp/{int(time.time())}"

    os.makedirs(LOG_DIR, exist_ok=True)


    # ---------------------- Hyperparameter Sweep ----------------------
    EXPERIMENT_NAME = "LipAlphaAndCodeReg"

    HPARAMS = [1e-2, 1e-3]  

    # Load base specs once
    with open(SPECS_FILES_DIR / SPECS_BASE_FILE, "r") as f:
        base_specs = json.load(f)

    # safer start method for CUDA + multiprocessing
    set_start_method("spawn", force=True)

    processes = []
    for alpha in HPARAMS:
        for sigma in HPARAMS:
            override_specs = {
                "Network_specs": {
                    "lipschitz_layers": [-1],
                    "use_lipschitz_normalized_layers": True
                },
                "lipschitz_alpha": alpha,
                "code_reg_lambda": sigma
            }

            # Validate overrides
            check_override_specs_validity(override_specs, base_specs)
            specs = deep_update(base_specs.copy(), override_specs)

            p = Process(
                target=run_single_training,
                args=(alpha, sigma, specs, LOG_DIR, EXPERIMENT_NAME, NUM_WORKERS)
            )
            p.start()

            processes.append(p)
            
            logger.info("Waiting ...")
            time.sleep(10)

    # wait for all to finish
    for p in processes:
        p.join()

    print("All trainings completed.")