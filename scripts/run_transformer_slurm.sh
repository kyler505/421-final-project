#!/bin/bash
#SBATCH --job-name=clinicalbert-train
#SBATCH --output=/scratch/user/kcao/csce421-final-project/logs/clinicalbert_train_%j.out
#SBATCH --error=/scratch/user/kcao/csce421-final-project/logs/clinicalbert_train_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/scratch/user/kcao/.conda/envs/tempdata/bin/python"
TRAIN_MANIFEST="${PROJECT_ROOT}/data/processed/combined_manifest.json"
MODEL_NAME="/scratch/user/kcao/csce421-final-project/models/pretrained/Bio_ClinicalBERT"
OUTPUT_DIR="/scratch/user/kcao/csce421-final-project/models/transformer_clinicalbert_combined"
MANIFEST_OUT="${OUTPUT_DIR}/run_manifest.json"

mkdir -p /scratch/user/kcao/csce421-final-project/logs

if [[ ! -d "${MODEL_NAME}" ]]; then
  echo "Error: missing pretrained checkpoint at ${MODEL_NAME}" >&2
  exit 1
fi
if [[ ! -f "${TRAIN_MANIFEST}" ]]; then
  echo "Error: missing train manifest at ${TRAIN_MANIFEST}" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HOME="/scratch/user/kcao/hf-cache"
export TRANSFORMERS_CACHE="/scratch/user/kcao/hf-cache"
export HUGGINGFACE_HUB_CACHE="/scratch/user/kcao/hf-cache/hub"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE"

"${PYTHON_BIN}" "${PROJECT_ROOT}/src/train_transformer.py" \
  --train-manifest "${TRAIN_MANIFEST}" \
  --output "${OUTPUT_DIR}" \
  --model_name "${MODEL_NAME}" \
  --epochs 3 \
  --batch_size 8 \
  --max_length 128 \
  --learning_rate 2e-5 \
  --manifest "${MANIFEST_OUT}"
