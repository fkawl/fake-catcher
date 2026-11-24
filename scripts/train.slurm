#!/usr/bin/env bash
#SBATCH --job-name=fake-catcher
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.log
set -euo pipefail
# Submit from project root. Add department-specific partition/account if required.
cd "$SLURM_SUBMIT_DIR"
bash scripts/train_linux.sh
