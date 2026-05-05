#!/usr/bin/env bash
#SBATCH --job-name=icd-train-baseline
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/icd-train-baseline-%j.out
#SBATCH --error=logs/icd-train-baseline-%j.err

set -euo pipefail

# Navigation
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"

# Conda environment activation
source $(conda info --base)/etc/profile.d/conda.sh
conda activate final
PYTHON_BIN=$(which python)

# Paths
TRAIN_CSV="data/phase1_weakly_labeled.csv"
MODEL_OUT="models/baseline_model.joblib"
SUMMARY_OUT="models/baseline_summary.json"

mkdir -p logs models

echo "Starting Model A (Baseline) training..."
# Train the Baseline (Model A)
# --no-embeddings: Skips the transformer part for now
# --no-self-training: Skips the iterative pseudo-labeling for the first baseline run
"$PYTHON_BIN" -m src.train \
    --train "$TRAIN_CSV" \
    --output "$MODEL_OUT" \
    --summary-output "$SUMMARY_OUT" \
    --no-embeddings \
    --no-self-training

echo "Baseline training complete. Model saved to $MODEL_OUT"
echo "Check models/baseline_summary.json for metrics."
