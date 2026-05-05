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
SSL_TEACHER_MODE="${SSL_TEACHER_MODE:-tfidf}"
TEACHER_TAG="${SSL_TEACHER_MODE}"
RUN_TAG="${RUN_TAG:-${SLURM_JOB_ID:-manual}-${TEACHER_TAG}}"
RUN_DIR="${RUN_DIR:-$PROJECT_DIR/runs/$RUN_TAG}"
TRAIN_CSV="${TRAIN_CSV:-$PROJECT_DIR/data/raw/train_data-text_and_labels.csv}"
NOTES_CSV_GZ="${NOTES_CSV_GZ:-/home/kcao/projects/csce421-final-project/data/raw/NOTEEVENTS.csv.gz}"
UNLABELED_CSV="${UNLABELED_CSV:-$PROJECT_DIR/data/processed/unlabeled_mimic.csv}"
MODEL_OUT="${MODEL_OUT:-$RUN_DIR/models/model.joblib}"
SUMMARY_OUT="${SUMMARY_OUT:-$RUN_DIR/models/training_summary.json}"
PRED_DIR="${PRED_DIR:-$RUN_DIR/predictions}"
DEBUG_DIR="${DEBUG_DIR:-$RUN_DIR/debug}"
MAX_UNLABELED="${MAX_UNLABELED:-150}"
UNLABELED_LIMIT="${UNLABELED_LIMIT:-3000}"
REPLACE_NUMBERS="${REPLACE_NUMBERS:-1}"
FIXED_ENSEMBLE_THRESHOLD="${FIXED_ENSEMBLE_THRESHOLD:-0.43}"
SSL_ROUNDS="${SSL_ROUNDS:-1}"
SSL_POSITIVE_CONFIDENCE="${SSL_POSITIVE_CONFIDENCE:-0.95}"
SSL_NEGATIVE_CONFIDENCE="${SSL_NEGATIVE_CONFIDENCE:-0.05}"
SSL_PSEUDO_WEIGHT="${SSL_PSEUDO_WEIGHT:-0.2}"
SSL_RANK_MODE="${SSL_RANK_MODE:-0}"
SSL_TEACHER_MODEL="${SSL_TEACHER_MODEL:-}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-$PROJECT_DIR/models/pretrained/Bio_ClinicalBERT}"
NO_EMBEDDINGS="${NO_EMBEDDINGS:-0}"
PUBLISH_ROOT_OUTPUTS="${PUBLISH_ROOT_OUTPUTS:-0}"

if [ "$SSL_TEACHER_MODE" = "embedding" ] && [ -z "$SSL_TEACHER_MODEL" ]; then
  SSL_TEACHER_MODEL="$EMBEDDING_MODEL"
fi

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/models" "$PROJECT_DIR/outputs" "$PROJECT_DIR/data/processed" "$RUN_DIR/models" "$PRED_DIR" "$DEBUG_DIR"
cd "$PROJECT_DIR"
ln -sfn "$RUN_DIR" "$PROJECT_DIR/latest-run-$RUN_TAG"

module purge
module load GCC/12.3.0 OpenMPI/4.1.5 scikit-learn/1.6.1
if [ -n "$SSL_TEACHER_MODEL" ] || [ "$NO_EMBEDDINGS" = "0" ]; then
  module load PyTorch/2.1.2-CUDA-12.1.1 Transformers/4.55.0
fi

PYTHON="${PYTHON:-python}"
"$PYTHON" - <<'PY'
import sklearn
print("sklearn", sklearn.__version__)
PY

"$PYTHON" -m src.prepare_unlabeled \
  --notes "$NOTES_CSV_GZ" \
  --output "$UNLABELED_CSV" \
  --limit "$UNLABELED_LIMIT"

TRAIN_ARGS=(
  --train "$TRAIN_CSV"
  --unlabeled "$UNLABELED_CSV"
  --output "$MODEL_OUT"
  --summary-output "$SUMMARY_OUT"
  --max-unlabeled "$MAX_UNLABELED"
  --ssl-rounds "$SSL_ROUNDS"
  --ssl-positive-confidence "$SSL_POSITIVE_CONFIDENCE"
  --ssl-negative-confidence "$SSL_NEGATIVE_CONFIDENCE"
  --ssl-pseudo-weight "$SSL_PSEUDO_WEIGHT"
  --ssl-rank-mode
  --ssl-teacher-mode "$SSL_TEACHER_MODE"
  --fixed-plan-weights
  --fixed-ensemble-threshold "$FIXED_ENSEMBLE_THRESHOLD"
)
if [ "$REPLACE_NUMBERS" != "0" ]; then
  TRAIN_ARGS=(--replace-numbers "${TRAIN_ARGS[@]}")
fi
if [ "$NO_EMBEDDINGS" = "0" ]; then
  TRAIN_ARGS=(--embedding-model "$EMBEDDING_MODEL" "${TRAIN_ARGS[@]}")
else
  TRAIN_ARGS=(--no-embeddings "${TRAIN_ARGS[@]}")
fi
if [ -n "$SSL_TEACHER_MODEL" ]; then
  TRAIN_ARGS=(--ssl-teacher-embedding-model "$SSL_TEACHER_MODEL" "${TRAIN_ARGS[@]}")
fi
"$PYTHON" -m src.train "${TRAIN_ARGS[@]}"

for split in test01 test02 test03; do
  pred_out="$PRED_DIR/${split}-pred.csv"
  debug_out="$DEBUG_DIR/${split}-debug.csv"
  "$PYTHON" -m src.predict \
    --model "$MODEL_OUT" \
    --input "$PROJECT_DIR/data/raw/${split}_text_only.csv" \
    --output "$pred_out" \
    --debug-output "$debug_out"
done

if [ "$PUBLISH_ROOT_OUTPUTS" != "0" ]; then
  for split in test01 test02 test03; do
    cp "$PRED_DIR/${split}-pred.csv" "$PROJECT_DIR/${split}-pred.csv"
    cp "$DEBUG_DIR/${split}-debug.csv" "$PROJECT_DIR/outputs/${split}-debug.csv"
  done
  cp "$SUMMARY_OUT" "$PROJECT_DIR/models/training_summary.json"
  cp "$MODEL_OUT" "$PROJECT_DIR/models/model.joblib"
fi
