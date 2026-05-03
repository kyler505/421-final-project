#!/bin/bash
#SBATCH --job-name=svm-eval
#SBATCH --output=/scratch/user/kevin.nguyen/csce421/final_project/421-final-project/logs/svm_eval_%j.out
#SBATCH --error=/scratch/user/kevin.nguyen/csce421/final_project/421-final-project/logs/svm_eval_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --chdir=.

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate final
PYTHON_BIN=$(which python)

mkdir -p /scratch/user/kevin.nguyen/csce421/final_project/421-final-project/logs

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "Starting SVM Cross-Validation Evaluation at $(date)"

# --- DATA CONFIGURATION ---
# We evaluate on the gold labels (course provided 20 rows)
TRAIN_CSV="${PROJECT_ROOT}/data/raw/train_data-text_and_labels.csv"
OUTPUT_JSON="${PROJECT_ROOT}/outputs/cv_svm_results.json"
FOLDS=5
# --------------------------

if [[ ! -f "${TRAIN_CSV}" ]]; then
  echo "Error: Training file not found at ${TRAIN_CSV}"
  exit 1
fi

echo "Running 5-fold CV for SVM on ${TRAIN_CSV}"
"${PYTHON_BIN}" -m src.run_eval \
    --train "${TRAIN_CSV}" \
    --mode svm \
    --folds "${FOLDS}" \
    --output "${OUTPUT_JSON}"

echo "----------------------------------------"
echo "Evaluation completed at $(date)"
echo "Results saved to ${OUTPUT_JSON}"
