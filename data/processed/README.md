# Training Data

This directory contains the training data manifests and pseudolabel CSV files for the CSCE 421 final project.

## Datasets

### 1. Baseline-Teacher (old) — Gold + Baseline Pseudolabels
- **Silver CSV**: `pseudolabels.csv` (7,640 rows, ~6 MB)
- **Gold CSV**: `data/raw/train_data-text_and_labels.csv` (MIMIC-III, NOT committed)
- **Manifest**: `combined_manifest.json` — references gold + baseline silver
- **Teacher model**: `models/baseline_model.pkl` (TF-IDF + logistic regression)
- **Total combined rows**: ~7,640

### 2. Transformer-Teacher (new) — Gold + ClinicalBERT Pseudolabels
- **Silver CSV**: `pseudolabels_transformer_teacher.csv` (665,583 rows, ~503 MB — too large for GitHub)
- **Gold CSV**: `data/raw/train_data-text_and_labels.csv` (MIMIC-III, NOT committed)
- **Manifest**: `manifest_transformer_teacher.json` — references gold + transformer silver
- **Teacher model**: `models/baseline_model_combined_tf_teacher.pkl`
- **Total combined rows**: ~665,603 (20 gold + 665,583 silver)

## Getting the Transformer-Teacher Silver CSV

The transformer-teacher pseudolabels are too large for GitHub (503 MB raw, 176 MB gzipped). Two options:

### Option A: Download the gzipped file
The compressed version is available at: *(provide cloud link or share manually)*

Unzip and place at `data/processed/pseudolabels_transformer_teacher.csv`.

### Option B: Regenerate on Grace
Submit the existing sbatch file on Grace:
```bash
sbatch scripts/run_pseudolabel_transformer_teacher.sbatch
```
This will regenerate the pseudolabels using the fine-tuned ClinicalBERT checkpoint.

## Usage

To train with either dataset, pass the corresponding manifest to the training script:

```bash
# Baseline-teacher
python -m src.train_baseline --train-manifest data/processed/combined_manifest.json

# Transformer-teacher
python -m src.train_baseline --train-manifest data/processed/manifest_transformer_teacher.json
```
