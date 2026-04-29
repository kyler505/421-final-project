#!/bin/bash
#SBATCH --job-name=pseudolabel-mimic
#SBATCH --output=/scratch/user/kcao/csce421-final-project/logs/pseudolabel_%j.out
#SBATCH --error=/scratch/user/kcao/csce421-final-project/logs/pseudolabel_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=short

set -euo pipefail

PROJECT_ROOT="/home/kyler/projects/csce421-final-project"
MIMIC_CSV="/path/to/NOTEEVENTS.csv.gz"          # set to your Grace path
BASELINE_MODEL="${PROJECT_ROOT}/models/baseline_model.pkl"
OUTPUT_DIR="${PROJECT_ROOT}/data/processed"
CONFIDENCE=0.95
BATCH_SIZE=512
PYTHON_BIN="/scratch/user/kcao/.conda/envs/tempdata/bin/python"

mkdir -p /scratch/user/kcao/csce421-final-project/logs
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "Starting pseudolabeling at $(date)"
"${PYTHON_BIN}" scripts/pseudolabel_mimic.py \
  --mimic-csv "${MIMIC_CSV}" \
  --baseline-path "${BASELINE_MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --confidence "${CONFIDENCE}" \
  --batch-size "${BATCH_SIZE}" \
  --categories "Discharge summary"

echo "Completed at $(date)"
