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
# To train on the old pseudolabels (TF-IDF teacher), change to the appropriate manifest.
# e.g., TRAIN_MANIFEST="${PROJECT_ROOT}/data/processed/old_manifest.json"
# Here we default to the combined manifest (new gold + silver)
TRAIN_MANIFEST="${PROJECT_ROOT}/data/processed/combined_manifest.json"
OUTPUT_MODEL="${PROJECT_ROOT}/models/svm_combined.pkl"
MANIFEST_OUT="${PROJECT_ROOT}/models/svm_combined_run_manifest.json"

mkdir -p /scratch/user/kcao/csce421-final-project/logs

if [[ ! -f "${TRAIN_MANIFEST}" ]]; then
  echo "Error: missing train manifest at ${TRAIN_MANIFEST}" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "Starting SVM training at $(date)"

"${PYTHON_BIN}" "${PROJECT_ROOT}/src/train_svm.py" \
  --train-manifest "${TRAIN_MANIFEST}" \
  --output "${OUTPUT_MODEL}" \
  --manifest "${MANIFEST_OUT}"

echo "Completed at $(date)"
