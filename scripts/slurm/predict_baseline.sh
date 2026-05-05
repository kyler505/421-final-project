#!/usr/bin/env bash
#SBATCH --job-name=icd-predict-baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/icd-predict-baseline-%j.out
#SBATCH --error=logs/icd-predict-baseline-%j.err

set -euo pipefail

# Navigation
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export TRANSFORMERS_OFFLINE=1

# Conda environment activation
source $(conda info --base)/etc/profile.d/conda.sh
conda activate final
PYTHON_BIN=$(which python)

# Paths
MODEL_PATH="models/full_ensemble_v1.joblib"
# Inputs (adjust paths if they are in data/)
TEST1="test01_text_only.csv"
TEST2="test02_text_only.csv"
TEST3="test03_text_only.csv"

mkdir -p logs outputs

echo "Generating BASELINE-ONLY predictions for Test 01..."
"$PYTHON_BIN" -m src.predict \
    --model "$MODEL_PATH" \
    --input "$TEST1" \
    --output "outputs/test01-baseline.csv" \
    --component ssl \
    --negation-filter

echo "Generating BASELINE-ONLY predictions for Test 02..."
"$PYTHON_BIN" -m src.predict \
    --model "$MODEL_PATH" \
    --input "$TEST2" \
    --output "outputs/test02-baseline.csv" \
    --component ssl \
    --negation-filter

echo "Generating BASELINE-ONLY predictions for Test 03..."
"$PYTHON_BIN" -m src.predict \
    --model "$MODEL_PATH" \
    --input "$TEST3" \
    --output "outputs/test03-baseline.csv" \
    --component ssl \
    --negation-filter

echo "Baseline predictions complete. Files saved to outputs/ (baseline names)"
