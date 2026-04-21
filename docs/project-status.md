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

Local checks completed:

- Python compile check
- smoke tests (`10 passed`)
- baseline training on the provided 20-row labeled dataset
- baseline prediction generation for:
  - `test01_text_only.csv`
  - `test02_text_only.csv`
  - `test03_text_only.csv`

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
- transformer training path exists but has not yet been run on a local offline checkpoint
- no cross-validation / ablation / report figures yet
- no MIMIC-III weak-supervision pipeline yet

## Recommended next steps

1. improve the baseline with stronger tokenization / n-gram settings
2. run a domain-specific transformer from a local checkpoint
3. decide whether to use MIMIC-III for weak supervision or extra unlabeled data
4. add evaluation and experiments for the final report
