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
EMBEDDING_MODEL="${EMBEDDING_MODEL:-$PROJECT_DIR/models/pretrained/Bio_ClinicalBERT}"
MODEL_OUT="${MODEL_OUT:-$PROJECT_DIR/models/model.joblib}"
SUMMARY_OUT="${SUMMARY_OUT:-$PROJECT_DIR/models/training_summary.json}"
SWEEP_OUT="${SWEEP_OUT:-$PROJECT_DIR/models/sweep_summary.json}"
MAX_UNLABELED="${MAX_UNLABELED:-500}"
UNLABELED_LIMIT="${UNLABELED_LIMIT:-20000}"

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/models" "$PROJECT_DIR/outputs" "$PROJECT_DIR/data/processed"
cd "$PROJECT_DIR"

module purge
module load GCC/13.3.0 OpenMPI/5.0.3 PyTorch/2.6.0 Transformers/4.55.0 scikit-learn/1.6.1

PYTHON="${PYTHON:-python}"
"$PYTHON" - <<'PY'
import sklearn, torch, transformers
print("sklearn", sklearn.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
PY

"$PYTHON" -m src.prepare_unlabeled \
  --notes "$NOTES_CSV_GZ" \
  --output "$UNLABELED_CSV" \
  --limit "$UNLABELED_LIMIT"

"$PYTHON" -m src.sweep \
  --train "$TRAIN_CSV" \
  --unlabeled "$UNLABELED_CSV" \
  --include-ssl \
  --max-unlabeled "$MAX_UNLABELED" \
  --output "$SWEEP_OUT"

"$PYTHON" -m src.train \
  --train "$TRAIN_CSV" \
  --unlabeled "$UNLABELED_CSV" \
  --max-unlabeled "$MAX_UNLABELED" \
  --embedding-model "$EMBEDDING_MODEL" \
  --fixed-plan-weights \
  --output "$MODEL_OUT" \
  --summary-output "$SUMMARY_OUT"

for split in test01 test02 test03; do
  "$PYTHON" -m src.predict \
    --model "$MODEL_OUT" \
    --input "$PROJECT_DIR/data/raw/${split}_text_only.csv" \
    --output "$PROJECT_DIR/${split}-pred.csv" \
    --debug-output "$PROJECT_DIR/outputs/${split}-debug.csv"
done
