#!/bin/bash

#SBATCH --job-name=lip_pen_sweep                                                           # Your Job Name
#SBATCH --nodes=1                                                                       # Number of Nodes desired e.g 1 node
#SBATCH --time=00:10:00                                                                 # Walltime: Duration for the Job to run HH:MM:SS
#SBATCH --mail-user=davide.navarri@medunigraz.at                                        # Your Email address assigned for your job
#SBATCH --mail-type=ALL                                                                 # Receive an email for ALL Job Statuses
#SBATCH --error=/home/isilon/users/o_navarri/experiments/logs-train-temp/%x_%J.err      # The .error file name captures the stderr of the whole batch script.
#SBATCH --output=/home/isilon/users/o_navarri/experiments/logs-train-temp/%x_%J.out     # The .output file name captures the stdout of the whole batch script.
#SBATCH --gres=gpu:1                                                                    # Resources e.g. GPU A100
#SBATCH --cpus-per-gpu=6                                                                # CPU cores per GPU
#SBATCH --mem=16G                                                                       # default on sx138 is 10GB per CPU core
#SBATCH --partition=gpu                                                                 # Partition: 'gpu' or 'cpu'
#SBATCH --nodelist=sx138                                                                # whitelist of nodes to use. it uses any if commented out


# ------------------- Setup -------------------

cd "/home/gpfs/o_navarri/projects/deepcsdf-atria" # "$SLURM_SUBMIT_DIR"

# Conda environment
source /home/gpfs/o_navarri/software/miniconda3/etc/profile.d/conda.sh
conda activate deepsdfatriavenvpy312

# Cleaner Python logging: Forces Python to flush stdout/stderr immediately, Slurm buffers output aggressively, so logs get cluttered
export PYTHONUNBUFFERED=1

# Disable CUDA MPS for independent GPU processes, I don't use it anyway
unset CUDA_MPS_PIPE_DIRECTORY

# Pin to the Slurm-assigned GPU: I request only ONE gpu, and inside a Slurm job, GPUs are renumbered, so don't use physical GPU ID
export CUDA_VISIBLE_DEVICES=0

# Use expandable segments for stable multi-process PyTorch GPU memory usage, allocate GPU memory in growable segments, instead of many fixed chunks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True



# ------------------- Parameters -------------------
EXPERIMENT=training_sweeps/LipPenaltyNoSpectral
SPECS_BASE=specs_files/specs_deepsdfatria-base.json
PYTHON_SCRIPT=train.py
SLEEP_INTERVAL=120
SAFETY_MARGIN_MB=500
MEM_REQUIRED_MB=2000

LOG_DIR=/home/isilon/users/o_navarri/experiments/logs-train-temp/$SLURM_JOB_ID # log dir unique per run
mkdir -p "$LOG_DIR"



# ------------------- GPU Memory / Concurrency -------------------
GPU_ID=0
GPU_TOTAL_MB=$(nvidia-smi --id=$GPU_ID --query-gpu=memory.total --format=csv,noheader,nounits)
MAX_PARALLEL=$(( GPU_TOTAL_MB / MEM_REQUIRED_MB ))
(( MAX_PARALLEL > 20 )) && MAX_PARALLEL=20  # optional safety cap

# Function to get free GPU memory in MB
function free_mem_mb() {
    nvidia-smi --id=$GPU_ID --query-gpu=memory.free --format=csv,noheader,nounits
}

# Function to remove finished job PIDs from array
declare -a JOB_PIDS=()
function clean_jobs() {
    local newlist=()
    for pid in "${JOB_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            newlist+=("$pid")
        fi
    done
    JOB_PIDS=("${newlist[@]}")
}

# ------------------- Hyperparameter sweep -------------------
LIP_PENALTY=(1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7)

for alpha in "${LIP_PENALTY[@]}"; do

        override_specs=$(cat <<BIPBOP
{
"Network_specs": {
"lipschitz_layers": [-1]
}
"use_lipreg_loss": true,
"lipschitz_alpha": $alpha
}
BIPBOP
)

        while true; do
            clean_jobs
            free=$(free_mem_mb)
            running="${#JOB_PIDS[@]}"

            if (( free > MEM_REQUIRED_MB + SAFETY_MARGIN_MB && running < MAX_PARALLEL )); then
                logfile="$LOG_DIR/Lipalpha${alpha}.log"
                echo "Launching lip alpha = ${alpha} | Free GPU: ${free}MB | Running jobs: ${running}"

                # Launch python in background with output redirected
                python "$PYTHON_SCRIPT" \
                    --experiment_name "$EXPERIMENT" \
                    --train_mode "compose_specs_from_options" \
                    --specs_file_path "$SPECS_BASE" \
                    --override_specs "$override_specs" \
                    &> "$logfile" &

                JOB_PIDS+=($!)
                sleep 10
                break
            else
                echo "Waiting for free GPU memory: $(date)"
                sleep "$SLEEP_INTERVAL"
            fi
        done
    done
done

# Wait for all jobs to finish
wait
echo "All trainings completed at $(date)"

# # ------------------- Optional: summary email -------------------
# VERSIONS=()
# for logfile in "$LOG_DIR"/*.log; do
#     version=$(grep "TRAINING_DONE_VERSION=" "$logfile" | cut -d= -f2)
#     [[ -n "$version" ]] && VERSIONS+=("$version")
# done

# if (( ${#VERSIONS[@]} > 0 )); then
#     EMAIL_BODY="Experiment: $EXPERIMENT"$'\n\n'"Versions:"
#     for v in "${VERSIONS[@]}"; do
#         EMAIL_BODY+=$'\n - '"$v"
#     done
#     python send_email.py --subject "Training concurrently: all jobs done" --body "$EMAIL_BODY"
# fi