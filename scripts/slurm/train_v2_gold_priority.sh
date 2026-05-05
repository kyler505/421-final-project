#!/usr/bin/env bash
#SBATCH --job-name=icd-train-v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/icd-train-v2-%j.out
#SBATCH --error=logs/icd-train-v2-%j.err

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
TRAIN_CSV="data/phase1_weakly_labeled.csv"
MODEL_OUT="models/full_ensemble_v2.joblib"
SUMMARY_OUT="models/full_summary_v2.json"

mkdir -p logs models

echo "Starting Version 2 Training (High Gold Priority + Clinical Negatives)..."
"$PYTHON_BIN" -m src.train \
    --train "$TRAIN_CSV" \
    --output "$MODEL_OUT" \
    --summary-output "$SUMMARY_OUT" \
    --gold-weight 50.0 \
    --embedding-model "emilyalsentzer/Bio_ClinicalBERT"

echo "V2 Training complete. Model saved to $MODEL_OUT"
