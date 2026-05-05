#!/usr/bin/env bash
#SBATCH --job-name=icd-sweep
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/icd-sweep-%j.out
#SBATCH --error=logs/icd-sweep-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate final
PYTHON_BIN=$(which python)

mkdir -p logs models outputs

# ============================================================
# STEP 1: Train gold-priority model (TF-IDF only, ~1 min)
# ============================================================
echo "=== Training Gold-Priority Model ==="
"$PYTHON_BIN" -m src.train \
    --train "data/phase1_weakly_labeled.csv" \
    --output "models/gold_priority.joblib" \
    --summary-output "models/gold_priority_summary.json" \
    --no-embeddings \
    --no-self-training \
    --logistic-c 0.1 \
    --gold-weight 50.0

echo "=== Gold-Priority training complete ==="

# ============================================================
# STEP 2: Generate predictions at multiple thresholds
# ============================================================
for MODEL_NAME in "gold_priority" "gold_only"; do
    MODEL_PATH="models/${MODEL_NAME}.joblib"
    if [ ! -f "$MODEL_PATH" ]; then
        echo "Skipping $MODEL_NAME (model not found)"
        continue
    fi

    for THRESH in 0.30 0.40 0.50 0.60 0.70 0.80; do
        echo "--- $MODEL_NAME threshold=$THRESH ---"
        for TESTNUM in 01 02 03; do
            "$PYTHON_BIN" -m src.predict \
                --model "$MODEL_PATH" \
                --input "test${TESTNUM}_text_only.csv" \
                --output "outputs/test${TESTNUM}-${MODEL_NAME}-t${THRESH}.csv" \
                --threshold "$THRESH"
        done
        # Show prediction distribution for test01
        echo "test01 distribution at threshold=$THRESH:"
        cut -d, -f2 "outputs/test01-${MODEL_NAME}-t${THRESH}.csv" | sort | uniq -c
    done
done

# ============================================================
# STEP 3: Also try the original ensemble v1 at different thresholds
# ============================================================
if [ -f "models/full_ensemble_v1.joblib" ]; then
    for THRESH in 0.60 0.70 0.80; do
        echo "--- ensemble_v1 (baseline component) threshold=$THRESH ---"
        for TESTNUM in 01 02 03; do
            "$PYTHON_BIN" -m src.predict \
                --model "models/full_ensemble_v1.joblib" \
                --input "test${TESTNUM}_text_only.csv" \
                --output "outputs/test${TESTNUM}-ensemble-baseline-t${THRESH}.csv" \
                --component baseline \
                --threshold "$THRESH"
        done
        echo "test01 distribution at threshold=$THRESH:"
        cut -d, -f2 "outputs/test01-ensemble-baseline-t${THRESH}.csv" | sort | uniq -c
    done
fi

echo "=== SWEEP COMPLETE ==="
echo "Look at the prediction distributions above."
echo "The best threshold should produce roughly 25-35 ones out of 80 predictions for test01."
echo "Submit whichever set has ~30-35% positive rate."
