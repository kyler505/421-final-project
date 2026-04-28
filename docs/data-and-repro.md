# Data and Reproducibility Notes

## Course data shape

The scaffold is designed around the assignment files:

- training CSV: `row_id,text,label`
- test CSVs: `row_id,text`
- submission CSVs: `row_id,prediction`

## Data handling policy

This repository does **not** commit:

- course-provided CSVs
- MIMIC-III data
- trained model artifacts
- generated prediction CSVs

Reasons:

- MIMIC-III has credentialed-access restrictions
- assignment data should stay local
- keeping data out of git makes the repo safe to publish and easy to share

## Local workflow

Put local data here:

```text
data/raw/
```

Expected local files for the assignment:

```text
data/raw/train_data-text_and_labels.csv
data/raw/test01_text_only.csv
data/raw/test02_text_only.csv
data/raw/test03_text_only.csv
```

## Multi-shard training (optional)

To combine the course CSV with additional local processed shards (for example silver-labeled MIMIC exports under `data/processed/`), use a version-1 JSON manifest and pass `--train-manifest` to `train_baseline` or `train_transformer`. Schema: [data-manifest-schema.md](data-manifest-schema.md).

## Reproducible CV metrics

Stratified K-fold on the baseline (useful for the 20-example gold set):

```bash
python -m src.run_eval --train data/raw/train_data-text_and_labels.csv --folds 5 --output outputs/cv_baseline.json
```

Training scripts emit a `*_run_manifest.json` (baseline) or `run_manifest.json` inside the transformer output directory listing package versions and hyperparameters.

## Reproducing the current baseline

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train baseline:

```bash
python -m src.train_baseline \
  --train data/raw/train_data-text_and_labels.csv \
  --output models/baseline_model.pkl
```

Generate a submission CSV:

```bash
python -m src.predict \
  --mode baseline \
  --model models/baseline_model.pkl \
  --input data/raw/test01_text_only.csv \
  --output outputs/test01-pred.csv
```

Optional debug CSV:

```bash
python -m src.predict \
  --mode baseline \
  --model models/baseline_model.pkl \
  --input data/raw/test01_text_only.csv \
  --output outputs/test01-pred.csv \
  --debug-output outputs/test01-debug.csv \
  --probabilities
```

## Transformer path

The transformer scaffold is designed so you can later point it at a local offline model checkpoint, e.g. a Bio_ClinicalBERT directory on disk.

Example:

```bash
python -m src.train_transformer \
  --train data/raw/train_data-text_and_labels.csv \
  --output models/transformer_model \
  --model_name /path/to/local/Bio_ClinicalBERT
```

## Test command

```bash
PYTHONPATH=. pytest -q tests/test_smoke.py
```
