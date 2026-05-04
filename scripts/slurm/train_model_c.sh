#!/bin/bash
#SBATCH --job-name=train-model-c
#SBATCH --output=logs/train_model_c_%j.out
#SBATCH --error=logs/train_model_c_%j.err
#SBATCH --time=04:00:00
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
CLF_TYPE="logistic" # or svm
C_VALUE=1.0
TRAIN_CSV="data/raw/train.csv"
OUTPUT_MODEL="models/model_c_clinicalbert.pkl"

cd "${PROJECT_ROOT}"
mkdir -p logs models

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "Starting Model C training at $(date)"
"${PYTHON_BIN}" -m src.train_model_c \
    --train "${TRAIN_CSV}" \
    --output "${OUTPUT_MODEL}" \
    --model_name "${MODEL_NAME}" \
    --max_length "${MAX_LENGTH}" \
    --batch_size "${BATCH_SIZE}" \
    --clf_type "${CLF_TYPE}" \
    --c "${C_VALUE}" \
    --use_tfidf

echo "Completed at $(date)"
