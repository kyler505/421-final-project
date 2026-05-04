#!/bin/bash
#SBATCH --job-name=cv-model-c
#SBATCH --output=logs/cv_model_c_%j.out
#SBATCH --error=logs/cv_model_c_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"

# Activate the clinicalbert virtual environment
source ~/clinicalbert_env/bin/activate
PYTHON_BIN="python"

# Model C configuration
MODEL_NAME="emilyalsentzer/Bio_ClinicalBERT"
MAX_LENGTH=128
BATCH_SIZE=32
CLF_TYPE="logistic"
TRAIN_CSV="data/raw/train.csv"

cd "${PROJECT_ROOT}"
mkdir -p logs

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "Starting Model C CV at $(date)"
"${PYTHON_BIN}" -m src.run_model_c_cv \
    --train "${TRAIN_CSV}" \
    --model_name "${MODEL_NAME}" \
    --max_length "${MAX_LENGTH}" \
    --batch_size "${BATCH_SIZE}" \
    --clf_type "${CLF_TYPE}" \
    --n_splits 5 \
    --use_tfidf

echo "Completed at $(date)"
