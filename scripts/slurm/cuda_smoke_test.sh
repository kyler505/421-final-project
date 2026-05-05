#!/usr/bin/env bash
#SBATCH --job-name=cuda-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/cuda-smoke-%j.out
#SBATCH --error=logs/cuda-smoke-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SCRATCH:?SCRATCH is not set}/csce421-final-project}"
USE_PIP_TORCH="${USE_PIP_TORCH:-0}"
PIP_TORCH_VERSION="${PIP_TORCH_VERSION:-2.6.0}"
PIP_CUDA_TAG="${PIP_CUDA_TAG:-cu124}"
LOCAL_BERT_MODEL="${LOCAL_BERT_MODEL:-$PROJECT_DIR/models/pretrained/Bio_ClinicalBERT}"
export LOCAL_BERT_MODEL

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

module purge
module load GCC/13.3.0 OpenMPI/5.0.3 scikit-learn/1.6.1

if [ "$USE_PIP_TORCH" = "1" ]; then
  python -m pip install --user --upgrade pip
  python -m pip install --user --upgrade \
    --index-url "https://download.pytorch.org/whl/${PIP_CUDA_TAG}" \
    "torch==${PIP_TORCH_VERSION}+${PIP_CUDA_TAG}" \
    "torchvision==0.21.0+${PIP_CUDA_TAG}" \
    "torchaudio==2.6.0+${PIP_CUDA_TAG}"
  python -m pip install --user --upgrade numpy transformers
else
  module load PyTorch/2.1.2-CUDA-12.1.1 Transformers/4.55.0
fi

python -u <<'PY'
import os
from glob import glob
from pathlib import Path

import torch

from src.embeddings import get_embedding_encoder

print("stage", "imported_torch", flush=True)
print("torch", torch.__version__)
print("torch_cuda_version", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
print("cuda_visible_devices", os.environ.get("CUDA_VISIBLE_DEVICES"))
if torch.cuda.is_available():
    print("current_device", torch.cuda.current_device())
    print("device_name", torch.cuda.get_device_name(0))

candidate_paths: list[Path] = []
env_model = os.environ.get("LOCAL_BERT_MODEL", "")
if env_model:
    candidate_paths.append(Path(env_model).expanduser())
cache_roots = [
    Path.home() / ".cache" / "huggingface" / "hub" / "models--emilyalsentzer--Bio_ClinicalBERT" / "snapshots",
    Path.home() / ".cache" / "huggingface" / "transformers",
]
for root in cache_roots:
    if root.exists():
        candidate_paths.extend(Path(p) for p in glob(str(root / "**"), recursive=True))

model_path = next((path for path in candidate_paths if path.exists() and (path / "config.json").exists()), None)
if model_path is None:
    raise FileNotFoundError(
        "Could not locate a local Bio_ClinicalBERT checkpoint. "
        "Set LOCAL_BERT_MODEL to a cached directory or sync the model into the Grace job environment."
    )
print("model_path", str(model_path))

print("stage", "loading_encoder", flush=True)
encoder = get_embedding_encoder(
    str(model_path),
    local_files_only=True,
)
print("stage", "encoder_loaded", flush=True)
print("encoder_device", encoder.device)
print("encoder_hidden_size", encoder.hidden_size)
print("stage", "encoding", flush=True)
vecs = encoder.encode(["test sentence one", "test sentence two"], batch_size=2)
print("stage", "encoded", flush=True)
print("encoded_shape", tuple(vecs.shape))
print("encoded_dtype", str(vecs.dtype))
print("encoded_first_norm", float((vecs[0] ** 2).sum() ** 0.5))
PY
