#!/usr/bin/env bash
#SBATCH --job-name=icd-full-pipeline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/icd-full-pipeline-%j.out
#SBATCH --error=logs/icd-full-pipeline-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SCRATCH:?SCRATCH is not set}/csce421-final-project}"
TRAIN_CSV="${TRAIN_CSV:-$PROJECT_DIR/data/raw/train_data-text_and_labels.csv}"
NOTES_CSV_GZ="${NOTES_CSV_GZ:-/home/kcao/projects/csce421-final-project/data/raw/NOTEEVENTS.csv.gz}"
UNLABELED_CSV="${UNLABELED_CSV:-$PROJECT_DIR/data/processed/unlabeled_mimic.csv}"
MODEL_OUT="${MODEL_OUT:-$PROJECT_DIR/models/model.joblib}"
SUMMARY_OUT="${SUMMARY_OUT:-$PROJECT_DIR/models/training_summary.json}"
SWEEP_OUT="${SWEEP_OUT:-$PROJECT_DIR/models/sweep_summary.json}"
MAX_UNLABELED="${MAX_UNLABELED:-150}"
UNLABELED_LIMIT="${UNLABELED_LIMIT:-3000}"
REPLACE_NUMBERS="${REPLACE_NUMBERS:-1}"
WEIGHT_STEP="${WEIGHT_STEP:-0.1}"
THRESHOLD_STEP="${THRESHOLD_STEP:-0.01}"
SSL_ROUNDS="${SSL_ROUNDS:-1}"
SSL_POSITIVE_CONFIDENCE="${SSL_POSITIVE_CONFIDENCE:-0.95}"
SSL_NEGATIVE_CONFIDENCE="${SSL_NEGATIVE_CONFIDENCE:-0.05}"
SSL_PSEUDO_WEIGHT="${SSL_PSEUDO_WEIGHT:-0.2}"
SSL_RANK_MODE="${SSL_RANK_MODE:-0}"

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/models" "$PROJECT_DIR/outputs" "$PROJECT_DIR/data/processed"
cd "$PROJECT_DIR"

module purge
module load GCC/13.3.0 OpenMPI/5.0.3 scikit-learn/1.6.1

PYTHON="${PYTHON:-python}"
"$PYTHON" - <<'PY'
import sklearn
print("sklearn", sklearn.__version__)
PY

"$PYTHON" -m src.prepare_unlabeled \
  --notes "$NOTES_CSV_GZ" \
  --output "$UNLABELED_CSV" \
  --limit "$UNLABELED_LIMIT"

SWEEP_ARGS=(
  --train "$TRAIN_CSV"
  --unlabeled "$UNLABELED_CSV"
  --include-ssl
  --max-unlabeled "$MAX_UNLABELED"
  --weight-step "$WEIGHT_STEP"
  --threshold-step "$THRESHOLD_STEP"
  --ssl-rounds "$SSL_ROUNDS"
  --ssl-positive-confidence "$SSL_POSITIVE_CONFIDENCE"
  --ssl-negative-confidence "$SSL_NEGATIVE_CONFIDENCE"
  --ssl-pseudo-weight "$SSL_PSEUDO_WEIGHT"
  --model-output "$MODEL_OUT"
  --output "$SWEEP_OUT"
)
if [ "$REPLACE_NUMBERS" != "0" ]; then
  SWEEP_ARGS=(--replace-numbers "${SWEEP_ARGS[@]}")
fi
if [ "$SSL_RANK_MODE" != "0" ]; then
  SWEEP_ARGS=(--ssl-rank-mode "${SWEEP_ARGS[@]}")
fi
"$PYTHON" -m src.sweep "${SWEEP_ARGS[@]}"
cp "$SWEEP_OUT" "$SUMMARY_OUT"

for split in test01 test02 test03; do
  "$PYTHON" -m src.predict \
    --model "$MODEL_OUT" \
    --input "$PROJECT_DIR/data/raw/${split}_text_only.csv" \
    --output "$PROJECT_DIR/${split}-pred.csv" \
    --debug-output "$PROJECT_DIR/outputs/${split}-debug.csv"
done
