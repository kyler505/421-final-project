#!/usr/bin/env bash
#SBATCH --job-name=icd-train-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/icd-train-full-%j.out
#SBATCH --error=logs/icd-train-full-%j.err

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
MODEL_OUT="models/full_ensemble_v1.joblib"
SUMMARY_OUT="models/full_summary_v1.json"

mkdir -p logs models

echo "Starting Full Ensemble Training (Baseline + SSL + ClinicalBERT)..."
# This will use the weights cached in ~/.cache/huggingface
# Make sure you ran the download command on a login node first!
"$PYTHON_BIN" -m src.train \
    --train "$TRAIN_CSV" \
    --output "$MODEL_OUT" \
    --summary-output "$SUMMARY_OUT" \
    --embedding-model "emilyalsentzer/Bio_ClinicalBERT"

echo "Full Ensemble training complete. Model saved to $MODEL_OUT"
echo "Check $SUMMARY_OUT for the combined ensemble metrics."
