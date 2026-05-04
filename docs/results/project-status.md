# Project status

## Current recommendation

**Ship the tuned linear baseline.**
The transformer/ClinicalBERT path was useful for comparison, but it did not beat the tuned linear model.

## What is implemented

- baseline classifier: TF-IDF + Logistic Regression
- SVM / linear sweep comparison utilities
- transformer scaffold for ClinicalBERT / Bio_ClinicalBERT
- stratified evaluation helpers
- pseudolabeling workflow for gold + silver training
- Gradescope-ready submission CSVs at the repo root

## Current best-known linear config

- `max_features=5000`
- `ngram_range=(1, 2)`
- `sublinear_tf=True`
- `C=0.25`

Observed local validation: `accuracy=0.85`, `F1=0.8933`

## Transformer comparison

The validated ClinicalBERT run completed successfully on Grace, but the result was weaker than the tuned linear baseline:

- `accuracy=0.75`
- `F1=0.6667`

## Completed artifacts

- `test01-pred.csv`
- `test02-pred.csv`
- `test03-pred.csv`
- `models/baseline_model_sweepbest.pkl`
- combined baseline and transformer comparison artifacts under `models/` and `outputs/`

## Short conclusion

The repo is now organized around a simple decision:

- **baseline = ship**
- **transformer = comparison / report material**
