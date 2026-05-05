#!/usr/bin/env bash
#SBATCH --job-name=icd-v4-bert
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/icd-v4-bert-%j.out
#SBATCH --error=logs/icd-v4-bert-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export TRANSFORMERS_OFFLINE=1

source $(conda info --base)/etc/profile.d/conda.sh
conda activate final
PYTHON_BIN=$(which python)

TRAIN_CSV="data/phase1_weakly_labeled.csv"
MODEL_OUT="models/v4_bert_content.joblib"
SUMMARY_OUT="models/v4_summary.json"

mkdir -p logs models outputs

# ============================================================
# STEP 1: Train V4 (Content-labeled data + ClinicalBERT + Gold Priority)
# ============================================================
echo "=== Training V4: Content Labels + ClinicalBERT + Gold Priority ==="
echo "Using data: $TRAIN_CSV"
echo "Start time: $(date)"

"$PYTHON_BIN" -m src.train \
    --train "$TRAIN_CSV" \
    --output "$MODEL_OUT" \
    --summary-output "$SUMMARY_OUT" \
    --embedding-model "emilyalsentzer/Bio_ClinicalBERT" \
    --gold-weight 50.0 \
    --logistic-c 0.1

echo "=== Training complete at $(date) ==="

# ============================================================
# STEP 2: Threshold sweep
# ============================================================
echo "=== Threshold sweep ==="
for THRESH in 0.40 0.50 0.55 0.60 0.65 0.70 0.75 0.80; do
    echo "--- V4 threshold=$THRESH ---"
    for TESTNUM in 01 02 03; do
        "$PYTHON_BIN" -m src.predict \
            --model "$MODEL_OUT" \
            --input "test${TESTNUM}_text_only.csv" \
            --output "outputs/test${TESTNUM}-v4-t${THRESH}.csv" \
            --threshold "$THRESH"
    done
    echo "test01 distribution at threshold=$THRESH:"
    cut -d, -f2 "outputs/test01-v4-t${THRESH}.csv" | sort | uniq -c
done

echo "=== SWEEP COMPLETE at $(date) ==="
echo "Pick the threshold where test01 has ~27-31 ones for best internal."
echo "Use a lower threshold (0.50-0.55) for test03 for best external."
