#!/bin/bash
#SBATCH --job-name=pseudolabel-mimic
#SBATCH --output=/scratch/user/kcao/csce421-final-project/logs/pseudolabel_%j.out
#SBATCH --error=/scratch/user/kcao/csce421-final-project/logs/pseudolabel_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=short

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
MIMIC_CSV="${PROJECT_ROOT}/data/raw/NOTEEVENTS.csv.gz"
OUTPUT_DIR="${PROJECT_ROOT}/data/processed/improved_pseudo"
mkdir -p "${OUTPUT_DIR}"

# Available models from the user:
# 1. baseline_model.pkl
# 2. baseline_model_combined.pkl
# 3. baseline_model_combined_tf_teacher.pkl
# 4. svm_baseline_teacher.pkl
# 5. svm_model_baseline_teacher.pkl
# 6. svm_model_transformer_teacher.pkl (Recommended)

TEACHER_MODEL="${PROJECT_ROOT}/models/svm_model_transformer_teacher.pkl"
CONFIDENCE=0.85
UPSAMPLE_GOLD=5

PYTHON_BIN="/scratch/user/kcao/.conda/envs/tempdata/bin/python"

mkdir -p /scratch/user/kcao/csce421-final-project/logs
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "Step 1: Starting pseudolabeling with ${TEACHER_MODEL} at $(date)"
"${PYTHON_BIN}" -m src.tools.pseudolabel_mimic \
  --mimic-csv "${MIMIC_CSV}" \
  --baseline-path "${TEACHER_MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --confidence "${CONFIDENCE}" \
  --min-silver-rows 10000 \
  --min-silver-fraction 0.10 \
  --min-per-class 2000 \
  --batch-size 1024 \
  --categories "Discharge summary"

echo "Step 2: Refining pseudolabels with filter_pseudo_labels.py at $(date)"
"${PYTHON_BIN}" src/tools/filter_pseudo_labels.py \
  --input "${OUTPUT_DIR}/pseudolabels.csv" \
  --output "${OUTPUT_DIR}/pseudolabels_filtered.csv" \
  --threshold_high 0.90 \
  --threshold_low 0.10 \
  --manifest "${OUTPUT_DIR}/manifest.json"

echo "Step 3: Training improved combined SVM at $(date)"
"${PYTHON_BIN}" -m src.train_svm \
  --train-manifest "${OUTPUT_DIR}/manifest.json" \
  --upsample-gold "${UPSAMPLE_GOLD}" \
  --output "${PROJECT_ROOT}/models/svm_model_improved_combined.pkl"

echo "Completed at $(date)"
