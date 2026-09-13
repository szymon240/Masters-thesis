#!/bin/bash
#SBATCH --job-name=FUSS_v2_Frams
#SBATCH --partition=obl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-17
#SBATCH --output=logs_slurm/%x_%A_%a.out
#SBATCH --error=logs_slurm/%x_%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs_slurm

python3 -u run_fuss_array.py --array-id "${SLURM_ARRAY_TASK_ID}"
