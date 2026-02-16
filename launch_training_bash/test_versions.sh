#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# ---- Parameters ----
VERSION_DIR=experiments/BatchSizeEffect
EXPERIMENT=BatchSizeEffect
PYTHON_SCRIPT=test.py
TEST_DATASET="train/data_fnames_train-20patients.json" #"test/data_fnames_test.json"
SLEEP_INTERVAL=2       # seconds
MAX_PARALLEL=1 # If computing LDDMM metric (which uses GPU acceleration), this may  be too many because GPU usage may peak a lot 
LOG_DIR=experiments/logs-test-temp
# --------------------
mkdir "$LOG_DIR"


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

echo "Experiment: $EXPERIMENT"

for dir in "$VERSION_DIR"/*/; do
    ver=${dir%/}
    ver="${ver##*/}"

    # num=${ver#version_}  # removes "version_" prefix
    # if (( num <= 2 || (num >= 10 && num <= 15) )); then
    #     echo "Skipping $ver"
    #     continue
    # fi

    while true; do
        clean_jobs

        running="${#JOB_PIDS[@]}"

        # Check if enough memory + below max jobs
        if (( running < MAX_PARALLEL )); then
            echo "Running $ver"

            logfile="$LOG_DIR/test_${ver}.log"

            python "$PYTHON_SCRIPT" \
                -e "$EXPERIMENT" \
                -v "$ver" \
                -od "$TEST_DATASET" \
                -nsamp 2048 \
                -N 300 \
                -lreg 1e-4 \
                -lddmm \
                -hauss \
                -chd \
                    &> "$logfile" &

            JOB_PIDS+=($!)
            sleep 2
            break
        else
            sleep "$SLEEP_INTERVAL"
        fi
    done
done


wait
echo "All testing completed at $(date)"

python "send_email.py" --subject "Testing concurrently: all jobs done"