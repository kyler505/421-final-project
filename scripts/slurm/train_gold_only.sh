#!/usr/bin/env bash
#SBATCH --job-name=icd-gold-only
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/icd-gold-only-%j.out
#SBATCH --error=logs/icd-gold-only-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate final
PYTHON_BIN=$(which python)

mkdir -p logs models outputs

# ============================================================
# MODEL 1: Gold-only (20 labels, no BERT, no SSL, strong regularization)
# ============================================================
echo "=== Training Gold-Only Model (20 labels, C=0.1) ==="
"$PYTHON_BIN" -m src.train \
    --train "train_data-text_and_labels.csv" \
    --output "models/gold_only.joblib" \
    --summary-output "models/gold_only_summary.json" \
    --no-embeddings \
    --no-self-training \
    --logistic-c 0.1

echo "=== Gold-Only Model Complete ==="

# Generate predictions immediately
for TESTNUM in 01 02 03; do
    echo "Predicting test${TESTNUM}..."
    "$PYTHON_BIN" -m src.predict \
        --model "models/gold_only.joblib" \
        --input "test${TESTNUM}_text_only.csv" \
        --output "outputs/test${TESTNUM}-gold-only.csv"
done

echo "=== Gold-Only predictions saved to outputs/ ==="

# ============================================================
# MODEL 2: Gold-priority (20k rows, no BERT, no SSL, gold weight=50)
# ============================================================
echo "=== Training Gold-Priority Model (20k rows, C=0.1, gold-weight=50) ==="
"$PYTHON_BIN" -m src.train \
    --train "data/phase1_weakly_labeled.csv" \
    --output "models/gold_priority.joblib" \
    --summary-output "models/gold_priority_summary.json" \
    --no-embeddings \
    --no-self-training \
    --logistic-c 0.1 \
    --gold-weight 50.0

echo "=== Gold-Priority Model Complete ==="

for TESTNUM in 01 02 03; do
    echo "Predicting test${TESTNUM}..."
    "$PYTHON_BIN" -m src.predict \
        --model "models/gold_priority.joblib" \
        --input "test${TESTNUM}_text_only.csv" \
        --output "outputs/test${TESTNUM}-gold-priority.csv"
done

echo "=== Gold-Priority predictions saved to outputs/ ==="
echo "Done! Submit the better set to Gradescope."
