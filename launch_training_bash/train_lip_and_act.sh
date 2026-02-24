#!/bin/bash

# ---- Parameters ----
EXPERIMENT=training_sweeps/LipAndAct
SPECS_BASE=specs_files/specs_deepsdfatria-base2.json
PYTHON_SCRIPT=train.py
SLEEP_INTERVAL=120       # seconds
SAFETY_MARGIN_MB=500   # safety
MEM_REQUIRED_MB=1500    
MAX_PARALLEL=16 
LOG_DIR=experiments/logs-train-temp
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
LIPSCHITZ_LAYERS=(
    '[0,1,2,3,4]'
    '[2,3,4]'
    '[3,4]'
    '[-1]'
)

ACTS=('Softplus' 'GELU' 'Tanh' 'SiLU')

for lip in "${LIPSCHITZ_LAYERS[@]}"; do
    for act in "${ACTS[@]}"; do
        
        # $() --> parentheses open a subshell, and stdout is captured to a variable,
        # cat alone reads from stdin and outputs to stdout
        # <<SOMETHING tells the shell: feed everything between here and the next SOMETHING as stdin to the command before it
        # so cat <<SOMETHING reads everything from stdin untile the next SOMETHING and sends it to stdout
        override_specs=$(cat <<BIPBOP
{
"Network_specs": {
"lipschitz_layers": $lip,
"activation" : "$act"
}
}
BIPBOP
)

        # echo "---- override_specs ----"
        # echo "$override_specs"
        # echo "------------------------"

        while true; do
            clean_jobs
            free=$(free_mem_mb)
            running="${#JOB_PIDS[@]}"

            if (( free > MEM_REQUIRED_MB + SAFETY_MARGIN_MB && running < MAX_PARALLEL )); then

                echo "Free GPU memory: ${free}MB | Running jobs: ${running} | Launching lip = ${lip} and act = ${act} ..."

                logfile="$LOG_DIR/lip${lip}_act${act}.log"

                # rename the process to show in nvidia-smi:  exec -a "name" python ...
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

# # # # echo "Cleaning up log directory..."
# # # # rm -rf "$LOG_DIR"
# # # # echo "Log directory removed."