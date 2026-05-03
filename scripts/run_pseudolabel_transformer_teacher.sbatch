#!/bin/bash
#SBATCH --job-name=tf-teacher-pseudolabel
#SBATCH --output=/scratch/user/kcao/csce421-final-project/logs/tf_teacher_pseudolabel_%j.out
#SBATCH --error=/scratch/user/kcao/csce421-final-project/logs/tf_teacher_pseudolabel_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=short
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_ROOT="/home/kcao/projects/csce421-final-project"
MIMIC_CSV="${PROJECT_ROOT}/data/raw/NOTEEVENTS.csv.gz"
OUTPUT_DIR="${PROJECT_ROOT}/data/processed"
PYTHON_BIN="/scratch/user/kcao/.conda/envs/tempdata/bin/python"
TEACHER_PATH="/scratch/user/kcao/csce421-final-project/models/transformer_clinicalbert_combined"

mkdir -p /scratch/user/kcao/csce421-final-project/logs
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HOME="/scratch/user/kcao/hf-cache"
export TRANSFORMERS_CACHE="/scratch/user/kcao/hf-cache"

echo "Starting transformer teacher pseudolabeling at $(date)"
echo "Teacher: ${TEACHER_PATH}"
echo "MIMIC: ${MIMIC_CSV}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/pseudolabel_mimic.py" \
  --mimic-csv "${MIMIC_CSV}" \
  --teacher-mode transformer \
  --teacher-path "${TEACHER_PATH}" \
  --teacher-batch-size 64 \
  --teacher-max-length 128 \
  --output-dir "${OUTPUT_DIR}" \
  --confidence 0.90 \
  --min-silver-rows 1000 \
  --min-silver-fraction 0.01 \
  --min-per-class 250 \
  --batch-size 512 \
  --categories "Discharge summary"

echo "Completed at $(date)"
