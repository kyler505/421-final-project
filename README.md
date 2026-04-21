# CSCE 421 Final Project - Clinical Note ICD Classification

Starter scaffold for a binary classifier that predicts whether a clinical-note sentence or text fragment contains ICD-codable medical information.

This is a **skeleton**, not a finished solution. It gives you a clean baseline path, a transformer path, and shared data-loading / prediction plumbing.

## Project structure

```text
csce421-final-project/
├── data/
│   ├── raw/                  # place train/test CSVs here
│   └── processed/
├── models/                   # saved artifacts
├── outputs/                  # prediction CSVs
├── report/                   # report assets / notes
├── docs/                     # project notes and reproducibility docs
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── predict.py
│   ├── train_baseline.py
│   ├── train_transformer.py
│   ├── utils.py
│   └── models/
│       ├── __init__.py
│       ├── baseline.py
│       └── transformer.py
├── tests/
│   └── test_smoke.py
├── requirements.txt
└── .gitignore
```

## Expected data shape

The scaffold assumes CSVs shaped like:

- training CSV header: `row_id,text,label`
- test CSV header: `row_id,text`
- submission CSV header: `row_id,prediction`

The loader uses light column inference, but those names are the default target format.

## What is included

- **Baseline model**: TF-IDF + Logistic Regression
- **Transformer scaffold**: Bio_ClinicalBERT-oriented wrapper + train entrypoint
- **Flexible CSV loader**: light inference for `row_id`, `text`, and `label`
- **Prediction CLI**: writes submission CSVs in `row_id,prediction` format
- **Optional debug CSV**: can also write `row_id,text,prediction[,probability]`
- **Smoke tests**: basic imports, loader behavior, and baseline save/load

## Setup

```bash
git clone https://github.com/kyler505/421-final-project.git
cd 421-final-project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want the transformer path later, uncomment/install the optional transformer dependencies in `requirements.txt`.

## Example CSV format

### Training CSV
Needs at least text and label columns; `row_id` is preserved if present.

```csv
row_id,text,label
0,"Patient presents with chest pain and elevated troponin.",1
1,"Patient was resting comfortably in bed.",0
```

### Test CSV
Needs `row_id` and a text-like column.

```csv
row_id,text
0,"History of uncontrolled hypertension and diabetes mellitus."
```

## Usage

### Train baseline

```bash
python -m src.train_baseline \
  --train data/raw/train.csv \
  --output models/baseline_model.pkl
```

### Predict with baseline

```bash
python -m src.predict \
  --mode baseline \
  --model models/baseline_model.pkl \
  --input data/raw/test.csv \
  --output outputs/test-pred.csv
```

With optional debug CSV:

```bash
python -m src.predict \
  --mode baseline \
  --model models/baseline_model.pkl \
  --input data/raw/test.csv \
  --output outputs/test-pred.csv \
  --debug-output outputs/test-debug.csv \
  --probabilities
```

### Train transformer scaffold

Use a local checkpoint path for fully offline work, or a model ID if you later decide to download it yourself.

```bash
python -m src.train_transformer \
  --train data/raw/train.csv \
  --output models/transformer_model \
  --model_name /path/to/local/Bio_ClinicalBERT \
  --epochs 3 \
  --batch_size 8
```

### Predict with transformer

```bash
python -m src.predict \
  --mode transformer \
  --model models/transformer_model \
  --input data/raw/test.csv \
  --output outputs/test-pred.csv
```

## Notes

- Default transformer max length is `128`.
- The transformer wrapper is import-safe even if `torch` / `transformers` are not installed.
- Submission output is the compact format: `row_id,prediction`.
- Use `--debug-output` if you also want a richer local inspection CSV.
- The transformer path is intentionally minimal; it is set up to be extended once you decide on evaluation, cross-validation, and checkpoint strategy.

## Tests

```bash
PYTHONPATH=. pytest -q tests/test_smoke.py
```
