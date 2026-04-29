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

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MIMIC_CSV="${PROJECT_ROOT}/data/raw/NOTEEVENTS.csv.gz"
BASELINE_MODEL="${PROJECT_ROOT}/models/baseline_model.pkl"
OUTPUT_DIR="${PROJECT_ROOT}/data/processed"
CONFIDENCE=0.90
MIN_SILVER_ROWS=1000
MIN_SILVER_FRACTION=0.01
MIN_PER_CLASS=250
BATCH_SIZE=512

# Stronger teacher route (optional): set TEACHER_MODE=transformer and point
# TEACHER_PATH at a fine-tuned gold-set checkpoint before submitting.
TEACHER_MODE="baseline"
TEACHER_PATH=""
TEACHER_BATCH_SIZE=""
TEACHER_MAX_LENGTH=""

PYTHON_BIN="/scratch/user/kcao/.conda/envs/tempdata/bin/python"

mkdir -p /scratch/user/kcao/csce421-final-project/logs
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "Starting pseudolabeling at $(date)"
cmd=(
  "${PYTHON_BIN}" scripts/pseudolabel_mimic.py
  --mimic-csv "${MIMIC_CSV}"
  --baseline-path "${BASELINE_MODEL}"
  --output-dir "${OUTPUT_DIR}"
  --confidence "${CONFIDENCE}"
  --min-silver-rows "${MIN_SILVER_ROWS}"
  --min-silver-fraction "${MIN_SILVER_FRACTION}"
  --min-per-class "${MIN_PER_CLASS}"
  --batch-size "${BATCH_SIZE}"
  --categories "Discharge summary"
)

if [[ "${TEACHER_MODE}" != "baseline" ]]; then
  cmd+=(--teacher-mode "${TEACHER_MODE}")
  if [[ -n "${TEACHER_PATH}" ]]; then
    cmd+=(--teacher-path "${TEACHER_PATH}")
  fi
  if [[ -n "${TEACHER_BATCH_SIZE}" ]]; then
    cmd+=(--teacher-batch-size "${TEACHER_BATCH_SIZE}")
  fi
  if [[ -n "${TEACHER_MAX_LENGTH}" ]]; then
    cmd+=(--teacher-max-length "${TEACHER_MAX_LENGTH}")
  fi
fi

"${cmd[@]}"

echo "Completed at $(date)"
