#!/usr/bin/env bash
#SBATCH --job-name=icd-data-prep
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/icd-data-prep-%j.out
#SBATCH --error=logs/icd-data-prep-%j.err

set -euo pipefail

# Navigation
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
echo "Running in $PROJECT_ROOT"

# Conda environment activation
source $(conda info --base)/etc/profile.d/conda.sh
conda activate final
PYTHON_BIN=$(which python)
echo "Using python: $PYTHON_BIN"

# Paths
# Based on user's Grace paths
MIMIC_DIR="/scratch/user/kevin.nguyen/csce421/mimiciii/1.4"
GOLD_LABELS="train_data-text_and_labels.csv"
OUTPUT_CSV="data/phase1_weakly_labeled.csv"

mkdir -p logs data

# Run the preparation script
"$PYTHON_BIN" -m src.data_preparation_phase1 \
    --mimic-dir "$MIMIC_DIR" \
    --gold-labels "$GOLD_LABELS" \
    --output "$OUTPUT_CSV" \
    --sample-size 10000 \
    --negative-sample-size 10000

echo "Data preparation complete. Output at $OUTPUT_CSV"
