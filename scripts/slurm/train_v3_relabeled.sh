#!/usr/bin/env bash
#SBATCH --job-name=icd-v3-full
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/icd-v3-full-%j.out
#SBATCH --error=logs/icd-v3-full-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate final
PYTHON_BIN=$(which python)

MIMIC_DIR="/scratch/user/kevin.nguyen/csce421/mimiciii/1.4"
GOLD_LABELS="train_data-text_and_labels.csv"
TRAIN_CSV="data/phase1_weakly_labeled.csv"

mkdir -p logs models outputs data

# ============================================================
# STEP 1: Regenerate training data with content-based labeling
# ============================================================
echo "=== STEP 1: Regenerating training data ==="
"$PYTHON_BIN" -m src.data_preparation_phase1 \
    --mimic-dir "$MIMIC_DIR" \
    --gold-labels "$GOLD_LABELS" \
    --output "$TRAIN_CSV" \
    --sample-size 10000 \
    --negative-sample-size 10000

echo "=== Data preparation complete ==="

# ============================================================
# STEP 2: Train model (TF-IDF only, gold weight=50, no BERT)
# ============================================================
echo "=== STEP 2: Training V3 model ==="
"$PYTHON_BIN" -m src.train \
    --train "$TRAIN_CSV" \
    --output "models/v3_content_labeled.joblib" \
    --summary-output "models/v3_summary.json" \
    --no-embeddings \
    --no-self-training \
    --logistic-c 0.1 \
    --gold-weight 50.0

echo "=== Training complete ==="

# ============================================================
# STEP 3: Generate predictions at multiple thresholds
# ============================================================
echo "=== STEP 3: Threshold sweep ==="
MODEL="models/v3_content_labeled.joblib"

for THRESH in 0.40 0.50 0.55 0.60 0.65 0.70 0.75 0.80; do
    echo "--- V3 threshold=$THRESH ---"
    for TESTNUM in 01 02 03; do
        "$PYTHON_BIN" -m src.predict \
            --model "$MODEL" \
            --input "test${TESTNUM}_text_only.csv" \
            --output "outputs/test${TESTNUM}-v3-t${THRESH}.csv" \
            --threshold "$THRESH"
    done
    echo "test01 distribution at threshold=$THRESH:"
    cut -d, -f2 "outputs/test01-v3-t${THRESH}.csv" | sort | uniq -c
done

# ============================================================
# STEP 4: Generate the "best guess" submission
# Target: ~27 ones out of 79 for test01 (~34% positive)
# ============================================================
echo "=== Generating final submission files ==="
# Use threshold 0.70 as default (adjust based on sweep output above)
for TESTNUM in 01 02 03; do
    "$PYTHON_BIN" -m src.predict \
        --model "$MODEL" \
        --input "test${TESTNUM}_text_only.csv" \
        --output "outputs/test${TESTNUM}-pred.csv" \
        --threshold 0.70
done

echo "=== ALL DONE ==="
echo "Check the threshold sweep output above to pick the best threshold."
echo "Submission files are in outputs/test##-pred.csv (using threshold 0.70)"
echo "If a different threshold looks better, copy those files instead."
