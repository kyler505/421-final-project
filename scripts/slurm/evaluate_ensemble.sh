#!/usr/bin/env bash
#SBATCH --job-name=icd-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/icd-eval-%j.out
#SBATCH --error=logs/icd-eval-%j.err

set -euo pipefail

# Navigation
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export TRANSFORMERS_OFFLINE=1

# Conda environment activation
source $(conda info --base)/etc/profile.d/conda.sh
conda activate final
PYTHON_BIN=$(which python)

# Paths
MODEL_PATH="models/full_ensemble_v1.joblib"
# Evaluating on the training set as a sanity check, 
# or a separate val.csv if you have one.
DATA_PATH="train_data-text_and_labels.csv"

mkdir -p logs

echo "Evaluating Ensemble on $DATA_PATH..."
"$PYTHON_BIN" -m src.evaluate \
    --model "$MODEL_PATH" \
    --data "$DATA_PATH" \
    --all-components

echo "Evaluation complete."
