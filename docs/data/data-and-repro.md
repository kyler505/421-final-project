# Data and Reproducibility Notes

## Canonical file layout

Course data and generated artifacts are expected in these locations:

- `data/raw/` — course-provided CSVs and any local gold training data
- `data/processed/` — generated silver data, manifests, and other derived training shards
- `models/` — trained pickles or checkpoint directories
- `outputs/` — inference and evaluation outputs

## Course file shapes

The project uses the following CSV conventions:

- training: `row_id,text,label`
- test: `row_id,text`
- submission: `row_id,prediction`

The root submission files for Gradescope are:

- `test01-pred.csv`
- `test02-pred.csv`
- `test03-pred.csv`

## What should stay out of git

Keep raw or generated data out of the source tree when possible:

- MIMIC-III source data
- course-provided CSVs if your submission policy forbids them
- large generated silver CSVs
- large model checkpoints unless you intentionally want them tracked

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Smoke test

```bash
PYTHONPATH=. pytest -q
```

## Reproducing the current baseline

Train the baseline:

```bash
python -m src.train_baseline \
  --train data/raw/train_data-text_and_labels.csv \
  --output models/baseline_model.pkl
```

Predict on a course test file:

```bash
python -m src.predict \
  --mode baseline \
  --model models/baseline_model.pkl \
  --input data/raw/test01_text_only.csv \
  --output test01-pred.csv
```

## Multi-shard training

If you want to combine gold and silver shards, pass a manifest to the training scripts:

```bash
python -m src.train_baseline --train-manifest data/processed/manifest_transformer_teacher.json
python -m src.train_transformer --train-manifest data/processed/manifest_transformer_teacher.json
```

See `data-manifest-schema.md` for the manifest contract.

## Current practical note

The repo's strongest completed submit candidate is the tuned linear baseline. The transformer path is useful for comparison, but it is not the default ship model.
