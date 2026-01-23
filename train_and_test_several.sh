#!/bin/bash

# ---- Parameters ----
EXPERIMENT=deepsdfatria_training_sweeps
SPECS_BASE=specs_files/specs_deepsdfatria-base.json
PYTHON_SCRIPT=train.py
SLEEP_INTERVAL=30       # seconds
SAFETY_MARGIN_MB=1000   # ~1 GB safety
MEM_REQUIRED_MB=2000    # with 89 scenes with 100000 points each all loaded in GPU,  2^14 points per scene, batches of 16 scenes, model with latent size 128, 512 x 7 layers, all lipschitz, --> memory usage tops out at 5108MiB /  24576MiB 
MAX_PARALLEL=10 
LOG_DIR=experiments/logs-temp
# --------------------

mkdir -p "$LOG_DIR"

# Function to get free GPU memory in MB
GPU_ID=1 
function free_mem_mb() {
    nvidia-smi \
      --id=$GPU_ID \
      --query-gpu=memory.free \
      --format=csv,noheader,nounits
}

# Clean job array function
function clean_jobs() {
    local newlist=()
    for pid in "${JOB_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            newlist+=("$pid")
        fi
    done
    JOB_PIDS=("${newlist[@]}")
}

declare -a JOB_PIDS=()


# ---- Hyperparameters to sweep ----
LIPSCHITZ_LAYERS=("[0,1,2,3,4,5,6]","[4,5,6]")
LR_WEIGHTS=(1e-3 5e-4)

for lip in "${LIPSCHITZ_LAYERS[@]}"; do
        for lr in "${LR_WEIGHTS[@]}"; do

            override_specs=$(cat <<EOF
{
  "Network_specs": {
    "lipschitz_layers": $lip
  },
  "lr_weights": $lr
}
EOF
)

            while true; do
                clean_jobs
                free=$(free_mem_mb)
                running="${#JOB_PIDS[@]}"

                if (( free > MEM_REQUIRED_MB + SAFETY_MARGIN_MB && running < MAX_PARALLEL )); then

                    logfile="$LOG_DIR/lip${lip}_lr${lr}.log"

                    python "$PYTHON_SCRIPT" \
                        --experiment_name "$EXPERIMENT" \
                        --train_mode "compose_specs_from_options" \
                        --specs_file_path "$BASE_SPECS" \
                        --override_specs "$override_specs" \
                        &> "$logfile" &

                    JOB_PIDS+=($!)
                    sleep 10
                    break
                else
                    sleep "$SLEEP_INTERVAL"
                fi
            done

        done
    
done

wait
echo "All trainings completed at $(date)"

# send email to myself with list of versions completed
VERSIONS=()
for logfile in "$LOG_DIR"/*.log; do
    version=$(grep "TRAINING_DONE_VERSION=" "$logfile" | cut -d= -f2)
    [[ -n "$version" ]] && VERSIONS+=("$version")
done

EMAIL_BODY=$'\n\nVersions:\n'
for v in "${VERSIONS[@]}"; do
    EMAIL_BODY+=$" - $v"
    EMAIL_BODY+=$'\n'
done

python "send_email.py" --subject "Training concurrently: all jobs done" --body "$EMAIL_BODY"

# echo "Cleaning up log directory..."
# rm -rf "$LOG_DIR"
# echo "Log directory removed."