#!/bin/bash
#SBATCH --job-name=pseudolabel-mimic
#SBATCH --output=logs/pseudolabel_%j.out
#SBATCH --error=logs/pseudolabel_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=normal

# Grace HPRC environment setup
# Adjust partition/time/mem based on availability and MIMIC size

set -euo pipefail

# ---- Config (edit these) ----
PROJECT_ROOT="/home/kyler/projects/csce421-final-project"
MIMIC_CSV="/path/to/NOTEEVENTS.csv.gz"          # <-- SET THIS
BASELINE_MODEL="${PROJECT_ROOT}/models/baseline_model.pkl"
OUTPUT_DIR="${PROJECT_ROOT}/data/processed"
CONFIDENCE=0.95
BATCH_SIZE=512
# --------------------------------

# Load Python (Grace environment)
module load python/3.11  # or your available Python module

# Enter project
cd "${PROJECT_ROOT}"

# Activate venv (create if missing)
if [ ! -d ".venv" ]; then
    python -m venv .venv
fi
source .venv/bin/activate

# Install deps (only needed first time)
pip install -q -r requirements.txt

# Create logs dir
mkdir -p logs

# Run pseudolabeling
echo "Starting pseudolabeling at $(date)"
python scripts/pseudolabel_mimic.py \
  --mimic-csv "${MIMIC_CSV}" \
  --baseline-path "${BASELINE_MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --confidence "${CONFIDENCE}" \
  --batch-size "${BATCH_SIZE}" \
  --categories "Discharge summary"

echo "Completed at $(date)"
