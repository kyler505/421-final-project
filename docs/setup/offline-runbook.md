# Offline runbook

This project is meant to run without internet once dependencies and checkpoints are in place.

## 1) Create the environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you plan to use the transformer path, make sure the required ML packages are already available locally.

## 2) Set a local checkpoint path

For transformer work, point the code at a local checkpoint directory:

```bash
export CSCE421_PRETRAINED_PATH=/path/to/local/Bio_ClinicalBERT
```

Do not rely on an online model download during the final run.

## 3) Run smoke tests

```bash
PYTHONPATH=. pytest -q
```

## 4) Train and predict

Baseline:

```bash
python -m src.train_baseline \
  --train data/raw/train_data-text_and_labels.csv \
  --output models/baseline_model.pkl

python -m src.predict \
  --mode baseline \
  --model models/baseline_model.pkl \
  --input data/raw/test01_text_only.csv \
  --output test01-pred.csv
```

Transformer:

```bash
python -m src.train_transformer \
  --train data/raw/train_data-text_and_labels.csv \
  --output models/transformer_model \
  --model_name "$CSCE421_PRETRAINED_PATH"
```

## 5) Submission files

Gradescope expects the root submission CSVs to be named:

- `test01-pred.csv`
- `test02-pred.csv`
- `test03-pred.csv`

Each file should contain only:

- `row_id`
- `prediction`

## 6) Canvas / zip contents

If you need to bundle the project for Canvas, include:

- `src/`
- `tests/`
- `docs/`
- `requirements.txt`
- any local checkpoint directory that the run depends on, if allowed
- the report PDF and source files

## 7) Practical guardrails

- keep caches and downloads on local disk or scratch, not inside a quota-limited home cache
- keep `PYTHONPATH` set if a script needs to import `src`
- avoid runtime network calls in the final submission path
