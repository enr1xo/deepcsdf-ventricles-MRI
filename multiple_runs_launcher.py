import json
from pprint import pprint
from multiprocessing import Process, set_start_method
import sys
import os
from pathlib import Path
from train import train, check_override_specs_validity, deep_update

# # ---------------------- Setup GPU memory ----------------------
# # optional: limit each process to a safe fraction of GPU memory
# torch.cuda.set_per_process_memory_fraction(1/40, device=0)
# torch.cuda.set_per_process_memory_growth(True)

SPECS_BASE_FILE = "specs_files/specs_deepsdfatria.json"

# If running inside Slurm, use $SLURM_JOB_ID, else fallback to PID or timestamp
slurm_job_id = os.environ.get("SLURM_JOB_ID", None)
if slurm_job_id is not None:
    log_dir = Path(f"/home/isilon/users/o_navarri/experiments/logs-train-temp/{slurm_job_id}")
else:
    import time
    log_dir = Path(f"/home/isilon/users/o_navarri/experiments/logs-train-temp/{int(time.time())}")


# ---------------------- Hyperparameter Sweep ----------------------
EXPERIMENT_NAME = "LipAlphaAndCodeReg"

HPARAMS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]  # same as your bash

# Load base specs once
with open(SPECS_BASE_FILE, "r") as f:
    base_specs = json.load(f)

# ---------------------- Function to launch a single training ----------------------
def run_single_training(alpha, sigma):

    logfile = log_dir / f"Lipalpha{alpha}_CodeReg{sigma}.log"
    with open(logfile, "w") as f:
        sys.stdout = f
        sys.stderr = f

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

        print(f"Starting training: alpha={alpha}, sigma={sigma}")
        pprint(specs)

        # call your existing train function
        train(specs)

# ---------------------- Launch all experiments concurrently ----------------------
if __name__ == "__main__":
    # safer start method for CUDA + multiprocessing
    set_start_method("spawn", force=True)

    processes = []
    for alpha in HPARAMS:
        for sigma in HPARAMS:
            p = Process(target=run_single_training, args=(alpha, sigma))
            p.start()
            processes.append(p)
            time.sleep(15)

    # wait for all to finish
    for p in processes:
        p.join()

    print("All trainings completed.")