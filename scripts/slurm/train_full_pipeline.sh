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
PIPELINE_MODE="${PIPELINE_MODE:-compare}"
SSL_TEACHER_MODE="${SSL_TEACHER_MODE:-tfidf}"
TEACHER_TAG="${SSL_TEACHER_MODE}"
RUN_TAG="${RUN_TAG:-${SLURM_JOB_ID:-manual}-${PIPELINE_MODE}-${TEACHER_TAG}}"
RUN_DIR="${RUN_DIR:-$PROJECT_DIR/runs/$RUN_TAG}"
TRAIN_CSV="${TRAIN_CSV:-$PROJECT_DIR/data/raw/train_data-text_and_labels.csv}"
NOTES_CSV_GZ="${NOTES_CSV_GZ:-/home/kcao/projects/csce421-final-project/data/raw/NOTEEVENTS.csv.gz}"
UNLABELED_CSV="${UNLABELED_CSV:-$PROJECT_DIR/data/processed/unlabeled_mimic.csv}"
MODEL_OUT="${MODEL_OUT:-$RUN_DIR/models/model.joblib}"
SUMMARY_OUT="${SUMMARY_OUT:-$RUN_DIR/models/training_summary.json}"
SWEEP_OUT="${SWEEP_OUT:-$RUN_DIR/models/sweep_summary.json}"
PSEUDO_MANIFEST_OUT="${PSEUDO_MANIFEST_OUT:-$RUN_DIR/models/pseudo_manifest.csv}"
PRED_DIR="${PRED_DIR:-$RUN_DIR/predictions}"
DEBUG_DIR="${DEBUG_DIR:-$RUN_DIR/debug}"
MAX_UNLABELED="${MAX_UNLABELED:-0}"
UNLABELED_LIMIT="${UNLABELED_LIMIT:-0}"
UNLABELED_PER_NOTE_LIMIT="${UNLABELED_PER_NOTE_LIMIT:-25}"
REPLACE_NUMBERS="${REPLACE_NUMBERS:-1}"
FIXED_ENSEMBLE_THRESHOLD="${FIXED_ENSEMBLE_THRESHOLD:-0.43}"
WEIGHT_STEP="${WEIGHT_STEP:-0.1}"
THRESHOLD_STEP="${THRESHOLD_STEP:-0.01}"
SSL_ROUNDS="${SSL_ROUNDS:-2}"
SSL_POSITIVE_CONFIDENCE="${SSL_POSITIVE_CONFIDENCE:-0.95}"
SSL_NEGATIVE_CONFIDENCE="${SSL_NEGATIVE_CONFIDENCE:-0.05}"
SSL_GOLD_WEIGHT="${SSL_GOLD_WEIGHT:-5.0}"
SSL_PSEUDO_WEIGHT="${SSL_PSEUDO_WEIGHT:-0.2}"
SSL_MAX_PSEUDO_PER_CLASS_PER_ROUND="${SSL_MAX_PSEUDO_PER_CLASS_PER_ROUND:-1000}"
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
module load GCC/12.3.0 OpenMPI/4.1.5
if [ -n "$SSL_TEACHER_MODEL" ] || [ "$NO_EMBEDDINGS" = "0" ]; then
  module load PyTorch/2.1.2-CUDA-12.1.1 Transformers/4.39.3
fi

PYTHON="${PYTHON:-python}"
"$PYTHON" - <<'PY'
import sklearn
print("sklearn", sklearn.__version__)
PY

"$PYTHON" -m src.prepare_unlabeled \
  --notes "$NOTES_CSV_GZ" \
  --output "$UNLABELED_CSV" \
  --max-sentences-per-note "$UNLABELED_PER_NOTE_LIMIT" \
  --limit "$UNLABELED_LIMIT"

if [ "$PIPELINE_MODE" = "compare" ]; then
  TRAIN_ARGS=(
    --train "$TRAIN_CSV"
    --unlabeled "$UNLABELED_CSV"
    --output "$MODEL_OUT"
    --summary-output "$SUMMARY_OUT"
    --max-unlabeled "$MAX_UNLABELED"
    --ssl-rounds "$SSL_ROUNDS"
    --ssl-positive-confidence "$SSL_POSITIVE_CONFIDENCE"
    --ssl-negative-confidence "$SSL_NEGATIVE_CONFIDENCE"
    --ssl-gold-weight "$SSL_GOLD_WEIGHT"
    --ssl-pseudo-weight "$SSL_PSEUDO_WEIGHT"
    --ssl-max-pseudo-per-class-per-round "$SSL_MAX_PSEUDO_PER_CLASS_PER_ROUND"
    --ssl-teacher-mode "$SSL_TEACHER_MODE"
    --fixed-plan-weights
    --fixed-ensemble-threshold "$FIXED_ENSEMBLE_THRESHOLD"
    --pseudo-manifest-output "$PSEUDO_MANIFEST_OUT"
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
elif [ "$PIPELINE_MODE" = "calibrate" ]; then
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
    --ssl-gold-weight "$SSL_GOLD_WEIGHT"
    --ssl-pseudo-weight "$SSL_PSEUDO_WEIGHT"
    --ssl-max-pseudo-per-class-per-round "$SSL_MAX_PSEUDO_PER_CLASS_PER_ROUND"
    --ssl-teacher-mode "$SSL_TEACHER_MODE"
    --model-output "$MODEL_OUT"
    --output "$SWEEP_OUT"
    --pseudo-manifest-output "$PSEUDO_MANIFEST_OUT"
    --calibration-mode
  )
  if [ "$REPLACE_NUMBERS" != "0" ]; then
    SWEEP_ARGS=(--replace-numbers "${SWEEP_ARGS[@]}")
  fi
  if [ "$NO_EMBEDDINGS" = "0" ]; then
    SWEEP_ARGS=(--embedding-model "$EMBEDDING_MODEL" "${SWEEP_ARGS[@]}")
  else
    SWEEP_ARGS=(--no-embeddings "${SWEEP_ARGS[@]}")
  fi
  if [ -n "$SSL_TEACHER_MODEL" ]; then
    SWEEP_ARGS=(--ssl-teacher-embedding-model "$SSL_TEACHER_MODEL" "${SWEEP_ARGS[@]}")
  fi
  "$PYTHON" -m src.sweep "${SWEEP_ARGS[@]}"
  cp "$SWEEP_OUT" "$SUMMARY_OUT"
else
  echo "Unknown PIPELINE_MODE: $PIPELINE_MODE" >&2
  exit 2
fi

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
