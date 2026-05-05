#!/usr/bin/env bash
#SBATCH --job-name=cuda-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/cuda-smoke-%j.out
#SBATCH --error=logs/cuda-smoke-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SCRATCH:?SCRATCH is not set}/csce421-final-project}"

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

module purge
module load GCC/13.3.0 OpenMPI/5.0.3 PyTorch/2.6.0 Transformers/4.55.0

python - <<'PY'
import os
import torch

from src.embeddings import get_embedding_encoder

print("torch", torch.__version__)
print("torch_cuda_version", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
print("cuda_visible_devices", os.environ.get("CUDA_VISIBLE_DEVICES"))
if torch.cuda.is_available():
    print("current_device", torch.cuda.current_device())
    print("device_name", torch.cuda.get_device_name(0))

encoder = get_embedding_encoder(
    "emilyalsentzer/Bio_ClinicalBERT",
    local_files_only=True,
)
print("encoder_device", encoder.device)
print("encoder_hidden_size", encoder.hidden_size)
vecs = encoder.encode(["test sentence one", "test sentence two"], batch_size=2)
print("encoded_shape", tuple(vecs.shape))
print("encoded_dtype", str(vecs.dtype))
print("encoded_first_norm", float((vecs[0] ** 2).sum() ** 0.5))
PY
