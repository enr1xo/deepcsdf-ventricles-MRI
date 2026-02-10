#!/bin/bash

#SBATCH --job-name=slurm_test                        # Your Job Name
#SBATCH --nodes=1                                       # Number of Nodes desired e.g 1 node
#SBATCH --time=00:10:00                                 # Walltime: Duration for the Job to run HH:MM:SS
#SBATCH --mail-user=davide.navarri@medunigraz.at          # Your Email address assigned for your job
#SBATCH --mail-type=ALL                                 # Receive an email for ALL Job Statuses
#SBATCH --error=/home/isilon/users/o_navarri/experiments/logs-train-temp/%x_%J.err                          # The .error file name captures the stderr of the whole batch script.
#SBATCH --output=/home/isilon/users/o_navarri/experiments/logs-train-temp/%x_%J.out                         # The .output file name captures the stdout of the whole batch script.
#SBATCH --gres=gpu:1                               # Resources e.g. GPU A100
#SBATCH --cpus-per-gpu=6                               # CPU cores per GPU
#SBATCH --mem=16G                                      # default on sx138 is 10GB per CPU core
#SBATCH --partition=gpu                                # Partition: 'gpu' or 'cpu'
#SBATCH --nodelist=sx138                               # whitelist of nodes to use. it uses any if commented out


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

EXPERIMENT=deepsdf-atria-training-single

SPECS_FILE_PATH=specs_files/specs_deepsdfatria.json

python train.py -e "$EXPERIMENT" -s "$SPECS_FILE_PATH" 