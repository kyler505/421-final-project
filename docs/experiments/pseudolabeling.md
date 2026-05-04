# Semi-Supervised Pseudolabeling

This document describes the MIMIC-III self-training workflow used in the project.

## Goal

Use a model trained on the small gold set to label additional MIMIC text, keep only high-confidence predictions, and turn them into silver training data.

## Workflow

1. Start with the gold training CSV.
2. Train a teacher model.
3. Chunk MIMIC notes into sentence-aware fragments.
4. Run teacher inference on each fragment.
5. Keep only confident predictions.
6. Write silver CSV plus a training manifest.
7. Train the downstream model on gold + silver.

## Teacher modes

The pipeline supports two teachers:

- `baseline` — TF-IDF + logistic regression
- `transformer` — ClinicalBERT / Bio_ClinicalBERT checkpoint

The transformer teacher is the stronger option when the baseline is too conservative.

## Current guardrails

To avoid the “zero silver rows” failure mode, the generator uses:

- a confidence threshold
- a minimum silver-row target
- a class-balanced fallback when coverage is too low
- per-class minimums when needed

## Common outputs

- `data/processed/pseudolabels.csv` or `data/processed/pseudolabels_transformer_teacher.csv`
- `data/processed/manifest.json` or `data/processed/manifest_transformer_teacher.json`

## Grace notes

For Grace runs:

- use the scratch-backed environment instead of runtime `pip install`
- keep Hugging Face caches on scratch
- submit through `/sw/local/bin/sbatch` when PATH is unreliable
- keep `PYTHONPATH` pointed at the repo root so `src` imports resolve

## Practical takeaway

The baseline-teacher path is mainly a fallback. The transformer-teacher path produced the useful silver set for the later combined runs.
