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

# Train on old manifest
TRAIN_MANIFEST_OLD="${PROJECT_ROOT}/data/processed/old_manifest.json"
OUTPUT_MODEL_OLD="${PROJECT_ROOT}/models/svm_old.pkl"
MANIFEST_OUT_OLD="${PROJECT_ROOT}/models/svm_old_run_manifest.json"

if [[ -f "${TRAIN_MANIFEST_OLD}" ]]; then
  echo "Training on old manifest: ${TRAIN_MANIFEST_OLD}"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/src/train_svm.py" \
    --train-manifest "${TRAIN_MANIFEST_OLD}" \
    --output "${OUTPUT_MODEL_OLD}" \
    --manifest "${MANIFEST_OUT_OLD}"
else
  echo "Warning: Old manifest not found at ${TRAIN_MANIFEST_OLD}"
fi

echo "----------------------------------------"

# Train on new combined manifest
TRAIN_MANIFEST_NEW="${PROJECT_ROOT}/data/processed/manifest.json"
OUTPUT_MODEL_NEW="${PROJECT_ROOT}/models/svm_combined.pkl"
MANIFEST_OUT_NEW="${PROJECT_ROOT}/models/svm_combined_run_manifest.json"

if [[ -f "${TRAIN_MANIFEST_NEW}" ]]; then
  echo "Training on new manifest: ${TRAIN_MANIFEST_NEW}"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/src/train_svm.py" \
    --train-manifest "${TRAIN_MANIFEST_NEW}" \
    --output "${OUTPUT_MODEL_NEW}" \
    --manifest "${MANIFEST_OUT_NEW}"
else
  echo "Warning: New manifest not found at ${TRAIN_MANIFEST_NEW}"
fi

echo "----------------------------------------"

# Evaluation step (requires a labeled test set)
# E.g. you can export TEST_CSV="/path/to/test.csv" before sbatch or set it here
TEST_CSV="${TEST_CSV:-${PROJECT_ROOT}/data/processed/test_gold.csv}"

if [[ -f "${TEST_CSV}" ]]; then
  echo "Evaluating models on ${TEST_CSV}..."
  
  if [[ -f "${OUTPUT_MODEL_OLD}" ]]; then
      echo "=== Evaluation: Old Model ==="
      "${PYTHON_BIN}" "${PROJECT_ROOT}/src/evaluate.py" \
          --model "${OUTPUT_MODEL_OLD}" \
          --input "${TEST_CSV}" \
          --mode svm
  fi
  
  if [[ -f "${OUTPUT_MODEL_NEW}" ]]; then
      echo "=== Evaluation: New Model ==="
      "${PYTHON_BIN}" "${PROJECT_ROOT}/src/evaluate.py" \
          --model "${OUTPUT_MODEL_NEW}" \
          --input "${TEST_CSV}" \
          --mode svm
  fi
else
  echo "Notice: Labeled test set not found at ${TEST_CSV}."
  echo "Skipping evaluation. You can run src/evaluate.py manually when you have the test set."
fi

echo "Completed at $(date)"
