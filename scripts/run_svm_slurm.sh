#!/bin/bash
#SBATCH --job-name=svm-train
#SBATCH --output=/scratch/user/kcao/csce421-final-project/logs/svm_train_%j.out
#SBATCH --error=/scratch/user/kcao/csce421-final-project/logs/svm_train_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=normal

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON_BIN="/scratch/user/kcao/.conda/envs/tempdata/bin/python"

mkdir -p /scratch/user/kcao/csce421-final-project/logs

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "Starting SVM training at $(date)"

# --- DATA CONFIGURATION ---
# SCENARIO 1: Baseline-Teacher (Old)
# TRAIN_MANIFEST="${PROJECT_ROOT}/data/processed/combined_manifest.json"
# OUTPUT_MODEL="${PROJECT_ROOT}/models/svm_baseline_teacher.pkl"

# SCENARIO 2: Transformer-Teacher (New)
TRAIN_MANIFEST="${PROJECT_ROOT}/data/processed/manifest_transformer_teacher.json"
OUTPUT_MODEL="${PROJECT_ROOT}/models/svm_transformer_teacher.pkl"
# --------------------------

MANIFEST_OUT="${OUTPUT_MODEL%.pkl}_run_manifest.json"

if [[ ! -f "${TRAIN_MANIFEST}" ]]; then
  echo "Error: Manifest not found at ${TRAIN_MANIFEST}"
  exit 1
fi

echo "Training SVM using manifest: ${TRAIN_MANIFEST}"
"${PYTHON_BIN}" "${PROJECT_ROOT}/src/train_svm.py" \
  --train-manifest "${TRAIN_MANIFEST}" \
  --output "${OUTPUT_MODEL}" \
  --manifest "${MANIFEST_OUT}"

echo "----------------------------------------"

# Evaluation step (requires a labeled test set)
TEST_CSV="${TEST_CSV:-${PROJECT_ROOT}/data/processed/test_gold.csv}"

if [[ -f "${TEST_CSV}" ]]; then
  echo "Evaluating model on ${TEST_CSV}..."
  "${PYTHON_BIN}" "${PROJECT_ROOT}/src/evaluate.py" \
      --model "${OUTPUT_MODEL}" \
      --input "${TEST_CSV}" \
      --mode svm
else
  echo "Notice: Labeled test set not found at ${TEST_CSV}. Skipping evaluation."
fi

echo "Completed at $(date)"
