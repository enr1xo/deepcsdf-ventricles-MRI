#!/bin/bash

# ---- Parameters ----
VERSION_DIR=experiments/training_sweeps/LipAndAct
PYTHON_SCRIPT=test.py
TEST_DATASET="train/data_fnames_train-20patients.json"
SLEEP_INTERVAL=60       # seconds
SAFETY_MARGIN_MB=500   # safety
MEM_REQUIRED_MB=6000    
MAX_PARALLEL=4 
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


for dir in "$VERSION_DIR"/*/; do
    ver=${dir%/}
    ver="${ver##*/}"

    if [ "$ver" != "version_34" ]; then
            
        while true; do
            clean_jobs
            free=$(free_mem_mb)
            running="${#JOB_PIDS[@]}"

            echo "Free GPU memory: ${free}MB | Running jobs: ${running}"

            # Check if enough memory + below max jobs
            if (( free > MEM_REQUIRED_MB + SAFETY_MARGIN_MB && running < MAX_PARALLEL )); then
                echo "Launching $ver"
                logfile="$LOG_DIR/$ver.log"

                python "$PYTHON_SCRIPT" -e "training_sweeps/LipAndAct" -v "$ver" -od "$TEST_DATASET" -cm &> "$logfile" &

                JOB_PIDS+=($!)
                sleep 10
                break
            else
                echo "Waiting for GPU memory..."
                sleep "$SLEEP_INTERVAL"
            fi
        done
    fi

done


wait
echo "All testing completed at $(date)"

python "send_email.py" --subject "Testing concurrently: all jobs done"