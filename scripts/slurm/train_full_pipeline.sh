#!/usr/bin/env bash
#SBATCH --job-name=icd-full-pipeline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/user/kcao/csce421-final-project-artifacts/logs/icd-full-pipeline-%j.out
#SBATCH --error=/scratch/user/kcao/csce421-final-project-artifacts/logs/icd-full-pipeline-%j.err

set -euo pipefail

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

PROJECT_DIR="${PROJECT_DIR:-${SCRATCH:?SCRATCH is not set}/csce421-final-project}"
JOB_ROOT="${JOB_ROOT:-${SCRATCH:?SCRATCH is not set}/csce421-final-project-artifacts}"
PIPELINE_MODE="${PIPELINE_MODE:-compare}"
SSL_TEACHER_MODE="${SSL_TEACHER_MODE:-tfidf}"
TEACHER_TAG="${SSL_TEACHER_MODE}"
RUN_TAG="${RUN_TAG:-${SLURM_JOB_ID:-manual}-${PIPELINE_MODE}-${TEACHER_TAG}}"
RUN_DIR="${RUN_DIR:-$JOB_ROOT/runs/$RUN_TAG}"
TRAIN_CSV="${TRAIN_CSV:-$JOB_ROOT/data/raw/train_data-text_and_labels.csv}"
NOTES_CSV_GZ="${NOTES_CSV_GZ:-$JOB_ROOT/data/raw/NOTEEVENTS.csv.gz}"
UNLABELED_CSV="${UNLABELED_CSV:-$JOB_ROOT/data/processed/unlabeled_mimic.csv}"
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
EMBEDDING_MODEL="${EMBEDDING_MODEL:-$JOB_ROOT/models/pretrained/Bio_ClinicalBERT}"
NO_EMBEDDINGS="${NO_EMBEDDINGS:-0}"
PUBLISH_ROOT_OUTPUTS="${PUBLISH_ROOT_OUTPUTS:-0}"

if [ "$SSL_TEACHER_MODE" = "embedding" ] && [ -z "$SSL_TEACHER_MODEL" ]; then
  SSL_TEACHER_MODEL="$EMBEDDING_MODEL"
fi

log "pipeline start mode=$PIPELINE_MODE teacher=$SSL_TEACHER_MODE run_tag=$RUN_TAG"
log "run_dir=$RUN_DIR"
mkdir -p "$JOB_ROOT/logs" "$JOB_ROOT/data/raw" "$JOB_ROOT/data/processed" "$JOB_ROOT/runs" "$RUN_DIR/models" "$PRED_DIR" "$DEBUG_DIR"
cd "$PROJECT_DIR"
ln -sfn "$RUN_DIR" "$PROJECT_DIR/latest-run-$RUN_TAG"

module purge
module load GCC/12.3.0 OpenMPI/4.1.5
if [ -n "$SSL_TEACHER_MODEL" ] || [ "$NO_EMBEDDINGS" = "0" ]; then
  module load PyTorch/2.1.2-CUDA-12.1.1 Transformers/4.39.3
fi
log "modules loaded"

PYTHON="${PYTHON:-python}"
log "checking sklearn runtime"
"$PYTHON" - <<'PY'
import sklearn
print("sklearn", sklearn.__version__)
PY

log "preparing unlabeled pool"
"$PYTHON" -m src.prepare_unlabeled \
  --notes "$NOTES_CSV_GZ" \
  --output "$UNLABELED_CSV" \
  --max-sentences-per-note "$UNLABELED_PER_NOTE_LIMIT" \
  --limit "$UNLABELED_LIMIT"
log "prepared unlabeled pool at $UNLABELED_CSV"

if [ "$PIPELINE_MODE" = "compare" ]; then
  log "running compare training"
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
  log "compare training finished"
elif [ "$PIPELINE_MODE" = "calibrate" ]; then
  log "running calibration sweep"
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
  log "calibration sweep finished"
else
  echo "Unknown PIPELINE_MODE: $PIPELINE_MODE" >&2
  exit 2
fi

log "writing predictions"
for split in test01 test02 test03; do
  pred_out="$PRED_DIR/${split}-pred.csv"
  debug_out="$DEBUG_DIR/${split}-debug.csv"
  "$PYTHON" -m src.predict \
    --model "$MODEL_OUT" \
    --input "$PROJECT_DIR/data/raw/${split}_text_only.csv" \
    --output "$pred_out" \
    --debug-output "$debug_out"
done
log "predictions finished"

if [ "$PUBLISH_ROOT_OUTPUTS" != "0" ]; then
  log "publishing root outputs"
  for split in test01 test02 test03; do
    cp "$PRED_DIR/${split}-pred.csv" "$JOB_ROOT/${split}-pred.csv"
    cp "$DEBUG_DIR/${split}-debug.csv" "$JOB_ROOT/${split}-debug.csv"
  done
  cp "$SUMMARY_OUT" "$JOB_ROOT/training_summary.json"
  cp "$MODEL_OUT" "$JOB_ROOT/model.joblib"
fi
log "pipeline complete"
