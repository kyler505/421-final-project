# CSCE 421 Final Project — Clinical Note ICD Classification

Binary text classification for the CSCE 421 final project: predict whether a clinical-note fragment contains ICD-codable medical information.

## Current recommendation

The current **submission candidate is the tuned linear baseline**. The transformer/ClinicalBERT path is kept as a comparison experiment, not the default ship model.

## Repo layout

```text
.
├── data/
├── docs/
├── models/
├── outputs/
├── scripts/
│   ├── dev/
│   └── slurm/
├── src/
│   └── tools/
├── tests/
├── test01-pred.csv
├── test02-pred.csv
└── test03-pred.csv
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. pytest -q
```

## Data conventions

- Training CSV: `row_id,text,label`
- Test CSV: `row_id,text`
- Submission CSV: `row_id,prediction`
- Root submission files must be named:
  - `test01-pred.csv`
  - `test02-pred.csv`
  - `test03-pred.csv`

## Typical commands

Train the baseline:

```bash
python -m src.train_baseline \
  --train data/raw/train_data-text_and_labels.csv \
  --output models/baseline_model.pkl
```

Run the tuned linear sweep on the gold set:

```bash
python -m src.sweep_linear \
  --train data/raw/train_data-text_and_labels.csv \
  --output outputs/sweep_linear_baseline.json
```

If you already have a combined manifest, you can point the same command at `--train-manifest` instead.

Predict on the course test files:

```bash
python -m src.predict \
  --mode baseline \
  --model models/baseline_model_sweepbest.pkl \
  --input data/raw/test01_text_only.csv \
  --output test01-pred.csv
```

## Documentation map

- `docs/data/data-and-repro.md` — data layout, reproducibility, and local setup
- `docs/data/data-manifest-schema.md` — manifest schema for multi-shard training
- `docs/experiments/pseudolabeling.md` — MIMIC pseudolabeling workflow
- `docs/models/primary-model.md` — primary model choice and offline checkpoint notes
- `docs/results/project-status.md` — current status, metrics, and recommended submit path
- `docs/setup/offline-runbook.md` — offline / Gradescope / Canvas packaging runbook
- `docs/archive/project-actions-log.md` — historical record of the main project steps
- `docs/course/CSCE421-FINAL-PROJECT-sp26-instructions.pdf` — assignment handout

## Notes

- Large generated artifacts stay out of the source tree when possible.
- `outputs/` contains comparison artifacts and evaluation CSVs.
- `models/` contains serialized models and checkpoints used for local reproducibility.
- For the latest status, start with `docs/results/project-status.md`.
