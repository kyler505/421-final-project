# Data and Reproducibility Notes

## Expected data shape

The scaffold is designed around CSVs with the following shapes:

- training CSV: `row_id,text,label`
- test CSVs: `row_id,text`
- submission CSVs: `row_id,prediction`

## Data handling policy

This repository does **not** commit:

- course-provided or private CSVs
- MIMIC-III data
- trained model artifacts
- generated prediction CSVs

Reasons:

- MIMIC-III has credentialed-access restrictions
- private datasets should stay local unless explicitly cleared for publication
- keeping data out of git makes the repo safe to share

## Local workflow

Put local data here:

```text
data/raw/
```

Example local files:

```text
data/raw/train.csv
data/raw/test.csv
```

You can also use assignment-specific names if preferred; the CLI commands just need the correct paths.

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
  --train data/raw/train.csv \
  --output models/baseline_model.pkl
```

Generate a submission CSV:

```bash
python -m src.predict \
  --mode baseline \
  --model models/baseline_model.pkl \
  --input data/raw/test.csv \
  --output outputs/test-pred.csv
```

Optional debug CSV:

```bash
python -m src.predict \
  --mode baseline \
  --model models/baseline_model.pkl \
  --input data/raw/test.csv \
  --output outputs/test-pred.csv \
  --debug-output outputs/test-debug.csv \
  --probabilities
```

## Transformer path

The transformer scaffold is designed so you can point it at a local offline model checkpoint, e.g. a Bio_ClinicalBERT directory on disk.

Example:

```bash
python -m src.train_transformer \
  --train data/raw/train.csv \
  --output models/transformer_model \
  --model_name /path/to/local/Bio_ClinicalBERT
```

## Test command

```bash
PYTHONPATH=. pytest -q tests/test_smoke.py
```
