# Project Status

Current repository status for the CSCE 421 final project scaffold.

## What is implemented

- project structure for code, docs, data, models, outputs, and tests
- baseline classifier: TF-IDF + Logistic Regression
- transformer scaffold aimed at Bio_ClinicalBERT / ClinicalBERT
- CSV loading with support for course file shapes:
  - training: `row_id,text,label`
  - test: `row_id,text`
- prediction writer that emits submission files in:
  - `row_id,prediction`
- optional debug prediction output with text and probabilities
- smoke tests covering imports, loader behavior, and prediction export helpers

## What has been tested

Local and Grace checks completed:

- Python compile check
- smoke tests (`10 passed`)
- baseline training on the provided 20-row labeled dataset
- baseline prediction generation for:
  - `test01_text_only.csv`
  - `test02_text_only.csv`
  - `test03_text_only.csv`
- ClinicalBERT transformer training on Grace (`COMPLETED`, exit `0:0`)
- transformer inference/evaluation on Grace (`COMPLETED`, exit `0:0`)
- pseudolabeling with baseline teacher — 7,620 silver rows
- pseudolabeling with transformer teacher — 665k silver rows
- retrained both baseline and transformer on gold + transformer-teacher silver
- four-way model comparison on all three test splits

## Available models

Three baseline models shipped in `models/`:

| Model | File | Trained on | Positives (test01/test02/test03) |
|---|---|---|---|
| Gold-only | `baseline_model.pkl` | 20 gold rows | 37 / 3911 / 57 |
| Old Combined | `baseline_model_combined.pkl` | gold + 7,620 baseline-teacher silver | 53 / 5739 / 161 |
| New Baseline (recommended) | `baseline_model_combined_tf_teacher.pkl` | gold + 665k transformer-teacher silver | 40 / 3870 / 75 |

Also tested but not shipped in repo:
- **New Transformer** (on Grace scratch) — trained on gold + 665k transformer-teacher silver — 39 / 3944 / 79

## Running predictions

Teammates can generate predictions with any model:

```bash
python -m src.predict \
  --mode baseline \
  --model models/baseline_model_combined_tf_teacher.pkl \
  --input data/raw/test01_text_only.csv \
  --output outputs/my_preds.csv
```

## Comparison summary

- New Baseline vs New Transformer: ~91% agreement
- New models are closer to gold-only than old combined was
- Old combined (baseline teacher) was over-permissive
- Transformer teacher pseudolabels were significantly better than baseline teacher

## Current baseline observations

- training set size: 20 rows
- label balance: 10 positive / 10 negative
- baseline training accuracy: 1.00
  - this should be treated as overfitting, not real quality
- baseline prediction rates:
  - test01: 37 / 79 positive
  - test02: 3911 / 7134 positive
  - test03: 57 / 168 positive

## Limitations right now

- baseline confidence is weak and clustered near 0.5
- very small labeled set means the baseline is mainly a sanity-check system
- transformer training path has now been run on Grace and is useful as a comparison experiment, not a default replacement
- no cross-validation / ablation / report figures yet
- no MIMIC-III weak-supervision pipeline yet

## Recommended next steps

1. decide whether to ship the combined baseline or keep the transformer as a comparison in the final report
2. add evaluation and experiment figures for the final report
3. optionally improve tokenization / n-gram settings if time remains
