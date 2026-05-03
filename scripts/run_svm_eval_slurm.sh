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

# --- EVALUATION CONFIGURATION ---
# The GOLD data to use for CV splitting and testing
GOLD_CSV="${PROJECT_ROOT}/data/raw/train_data-text_and_labels.csv"
FOLDS=5

# Function to run augmented CV evaluation
run_augmented_cv() {
    local augment_manifest=$1
    local output=$2
    local label=$3

    echo "Running Augmented CV for: ${label}"
    
    local cmd=(
        "${PYTHON_BIN}" -m src.run_eval
        --train "${GOLD_CSV}"
        --mode svm
        --folds "${FOLDS}"
        --output "${output}"
    )

    if [[ -f "${augment_manifest}" ]]; then
        cmd+=(--augment-manifest "${augment_manifest}")
    else
        echo "Warning: Augment manifest not found at ${augment_manifest}. Running Gold-only CV."
    fi

    "${cmd[@]}"
}

# SCENARIO 0: Baseline (Gold labels only, no silver augmentation)
OUT_GOLD="${PROJECT_ROOT}/outputs/cv_svm_gold_only.json"
run_augmented_cv "" "${OUT_GOLD}" "Gold-only (Baseline)"

echo "----------------------------------------"

# SCENARIO 1: SVM augmented with Baseline-Teacher Silver Data
AUG_BASE="${PROJECT_ROOT}/data/processed/manifest_silver_baseline.json"
OUT_BASE="${PROJECT_ROOT}/outputs/cv_svm_augmented_baseline.json"
run_augmented_cv "${AUG_BASE}" "${OUT_BASE}" "Baseline-Teacher Augmented"

echo "----------------------------------------"

# SCENARIO 2: SVM augmented with Transformer-Teacher Silver Data
AUG_TRANS="${PROJECT_ROOT}/data/processed/manifest_silver_transformer.json"
OUT_TRANS="${PROJECT_ROOT}/outputs/cv_svm_augmented_transformer.json"
run_augmented_cv "${AUG_TRANS}" "${OUT_TRANS}" "Transformer-Teacher Augmented"

echo "----------------------------------------"
echo "Evaluation completed at $(date)"
