# CSCE 421 Final Project - Clinical Note ICD Classification

Starter scaffold for a binary classifier that predicts whether a clinical-note sentence or text fragment contains ICD-codable medical information.

This is a **skeleton**, not a finished solution. It gives you a clean baseline path, a transformer path, shared data-loading / prediction plumbing, **and a semi-supervised pseudolabeling pipeline** for scaling with MIMIC-III.

## Project structure

```text
csce421-final-project/
├── data/
│   ├── raw/                  # place train/test CSVs here
│   └── processed/            # pseudolabels.csv + manifest.json go here
├── models/                   # saved artifacts (baseline_model.pkl, transformer checkpoints)
├── outputs/                  # prediction CSVs
├── report/                   # report assets / notes
├── scripts/
│   ├── pseudolabel_mimic.py  # MIMIC-III pseudolabel generator (self-training)
│   ├── run_pseudolabel_slurm.sh  # Slurm submission script for Grace HPRC
│   └── freeze_requirements.ps1
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── predict.py
│   ├── run_eval.py
│   ├── train_baseline.py
│   ├── train_transformer.py
│   ├── contracts.py
│   ├── manifest.py
│   ├── eval_metrics.py
│   ├── eval_cv.py
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

## What is included

- **Baseline model**: TF-IDF + Logistic Regression
- **Transformer scaffold**: Bio_ClinicalBERT-oriented wrapper + train entrypoint
- **Flexible CSV loader**: light inference for `row_id`, `text`, and `label`
- **Prediction CLI**: writes submission CSVs in `row_id,prediction` format
- **Optional debug CSV**: can also write `row_id,text,prediction[,probability]`
- **Smoke tests**: basic imports, loader behavior, and baseline save/load
- **Public contracts**: `src/contracts.py` fixes submission column names and validates binary predictions; `--mode` / `--backend` on `predict.py` are aliases
- **Run manifests**: JSON written after training (and optional `--write-manifest` on predict) for reproducibility
- **Evaluation**: `python -m src.run_eval` for stratified CV on the baseline; `src/eval_metrics.py` shared with `sweep_transformer`
- **Multi-shard training**: `--train-manifest` on train scripts + schema in [docs/data-manifest-schema.md](docs/data-manifest-schema.md)
- **Semi-supervised pipeline**: pseudolabel MIMIC-III with a gold-trained baseline, then fine-tune on gold + silver (see below)

## Operations and offline submission

- [docs/OFFLINE_RUNBOOK.md](docs/OFFLINE_RUNBOOK.md) — environment, pinning deps, train/predict, Canvas zip
- [docs/primary-model.md](docs/primary-model.md) — local HF checkpoint as primary artifact; `CSCE421_PRETRAINED_PATH` for defaults

## Setup

```bash
cd /home/kyler/csce421-final-project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want the transformer path later, uncomment/install the optional transformer dependencies in `requirements.txt`.

## Expected CSV format

### Training CSV
Needs at least text and label columns; `row_id` is preserved if present.

Example:

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

This writes:
- `models/baseline_model.pkl` — serialized TF-IDF + LogisticRegression
- `models/baseline_model_manifest.json` — RunManifest with CLI args + SHA256

### Evaluate baseline (stratified CV on labeled CSV)

```bash
python -m src.run_eval \
  --train data/raw/train_data-text_and_labels.csv \
  --folds 5 \
  --output outputs/cv_baseline.json
```

### Predict with baseline

```bash
python -m src.predict \
  --mode baseline \
  --model models/baseline_model.pkl \
  --input data/raw/test01_text_only.csv \
  --output outputs/test01-pred.csv
```

With optional debug CSV:

```bash
python -m src.predict \
  --mode baseline \
  --model models/baseline_model.pkl \
  --input data/raw/test01_text_only.csv \
  --output outputs/test01-pred.csv \
  --debug-output outputs/test01-debug.csv \
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
  --input data/raw/test01_text_only.csv \
  --output outputs/test01-pred.csv
```

---

## Semi-Supervised Pseudolabeling (MIMIC-III)

### Overview

This pipeline uses **self-training**: a gold-trained baseline model (TF-IDF + Logistic Regression) labels a large unlabeled corpus (MIMIC-III `NOTEEVENTS`), and only **high-confidence predictions** (default 0.95) are kept as silver data. The silver CSV is then merged with your gold training set for transformer fine-tuning.

**Why self-training over distant supervision?**
- Avoids noisy ICD-code heuristics that introduce label errors
- Respects the project's strict binary-label validation (0/1 only)
- Keeps the signal clean: only model-predicted fragments that cross a high confidence threshold become training data

## Setup

To create a fresh environment and install dependencies:

```bash
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

### 1. `scripts/pseudolabel_mimic.py`

**Purpose:** Chunk MIMIC notes into 128-word fragments (sentence-aware), run batched inference with a trained baseline, filter by confidence, and emit a manifest that unions gold + silver sources.

**Key flags:**

| Flag | Required? | Default | Description |
|------|-----------|---------|-------------|
| `--mimic-csv` | yes | — | Path to `NOTEEVENTS.csv` or `NOTEEVENTS.csv.gz` |
| `--baseline-path` | no | `models/baseline_model.pkl` (from config) | Trained baseline model |
| `--output-dir` | no | `data/processed/` | Where `pseudolabels.csv` + `manifest.json` are written |
| `--confidence` | no | `0.95` | Minimum `predict_proba` to include a silver fragment |
| `--max-words` | no | `128` (from `config.max_words_course`) | Fragment word limit per chunk |
| `--batch-size` | no | `256` | Inference batch size |
| `--categories` | no | `"Discharge summary"` | Comma-separated `CATEGORY` filter on NOTEEVENTS |
| `--sample` | no | (all) | Only process first N notes — useful for quick local tests |
| `--gold-csv` | no | `data/raw/train.csv` | Your existing gold training CSV |

**What it writes:**
- `data/processed/pseudolabels.csv` — `row_id,text,label` (label is the predicted class)
- `data/processed/manifest.json` — Multi-shard manifest unioning gold CSV + pseudolabels CSV with metadata (source, row_id offset, split tag)

**Chunking strategy:**
- Sentence tokenization via `re.split(r'[.!?]+', text)` (no NLTK dependency)
- Consecutive sentences are accumulated until `max_words` words are reached; overflow starts a new chunk
- 128-word cap matches the project's course-specified fragment size
- Each chunk inherits the parent note's `row_id` prefix plus a local chunk index (monotonic integer counter across the entire file, not string-concatenated, to match `src/contracts.py` integer ID expectations)

**Confidence filtering:**
- Uses `BaselineModel.predict_proba(X)` → probability of class `1`
- Keeps only fragments where `max(probas) >= --confidence`
- Applies the **most confident class** directly as the silver `label` (0 or 1)

**Manifest format:**
```json
{
  "version": "1.0",
  "shards": [
    {
      "path": "data/raw/train.csv",
      "source": "gold",
      "split": "train",
      "row_id_offset": 0,
      "label_column": "label"
    },
    {
      "path": "data/processed/pseudolabels.csv",
      "source": "silver",
      "split": "train",
      "row_id_offset": 1000000,
      "label_column": "label"
    }
  ]
}
```

### 2. `scripts/run_pseudolabel_slurm.sh`

**Purpose:** One-command Slurm submission for Grace HPRC. Sets up Python environment, installs deps, runs the pseudolabeler, and streams output to `logs/`.

**Prerequisites on Grace:**
- Python 3.11 module available (`module load python/3.11`)
- `NOTEEVENTS.csv.gz` located at the path you set in `MIMIC_CSV`
- Your gold `train.csv` already present in `data/raw/`
- Baseline model already trained and at `models/baseline_model.pkl`

**Configuration (edit the script before submitting):**

```bash
PROJECT_ROOT="/home/kyler/projects/csce421-final-project"
MIMIC_CSV="/path/to/NOTEEVENTS.csv.gz"   # <—— CHANGE THIS
BASELINE_MODEL="${PROJECT_ROOT}/models/baseline_model.pkl"
OUTPUT_DIR="${PROJECT_ROOT}/data/processed"
CONFIDENCE=0.95
BATCH_SIZE=512
```

**Resource requests (customize as needed):**

```bash
#SBATCH --time=02:00:00    # Increase for full MIMIC (~1.5–2 h)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=normal
```

**Lifecycle:**
1. Loads Python module
2. Creates/activates project `.venv`
3. Installs `requirements.txt` (first run only)
4. Creates `logs/` directory
5. Runs `python scripts/pseudolabel_mimic.py …`
6. Exit code propagates from Python; Slurm writes stdout/stderr to `logs/`

**Submission:**

```bash
cd /home/kyler/projects/csce421-final-project
sbatch scripts/run_pseudolabel_slurm.sh
```

**Monitor:**

```bash
squeue -u $USER               # job排队状态
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS,Node  # 完成后查阅
tail -f logs/pseudolabel_<JOBID>.out
```

### Full end-to-end workflow

```bash
# ── On your Mac (or any Linux workstation) ────────────────────────────────

# 1. Clone & set up
git clone git@github.com:kyler505/421-final-project.git
cd 421-final-project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Put your gold train/test CSVs in data/raw/
#    Expected headers: row_id,text,label  (train)  /  row_id,text  (test)

# 3. Train the gold-only baseline (needed for pseudolabeling)
python -m src.train_baseline \
  --train data/raw/train.csv \
  --output models/baseline_model.pkl

# 4. OPTIONAL: quick local pseudolabeling sanity-check on 100 notes
#    (Download a small NOTEEVENTS sample or use --sample 100 on a full copy)
python scripts/pseudolabel_mimic.py \
  --mimic-csv /path/to/NOTEEVENTS.csv.gz \
  --baseline-path models/baseline_model.pkl \
  --output-dir data/processed \
  --confidence 0.95 \
  --sample 100

# 5. Transfer data to Grace (via thinkpad-work hop or direct scp)
#    Using the grace-via-thinkpad-work skill / your preferred method:
#      - NOTEEVENTS.csv.gz
#      - The entire project repo (or push via GitHub & git pull on Grace)
#      - models/baseline_model.pkl

# ── On Grace (HPRC) ─────────────────────────────────────────────────────────

# 6. Pull latest from GitHub (if you pushed)
cd /home/kyler/projects/csce421-final-project
git pull origin main

# 7. Edit scripts/run_pseudolabel_slurm.sh → set MIMIC_CSV to the actual path
#    (e.g. /scratch/kyler/NOTEEVENTS.csv.gz)

# 8. Submit
sbatch scripts/run_pseudolabel_slurm.sh

# 9. Wait for completion (check squeue / sacct / tail -f logs/*.out)
#    Upon success you will have:
#      data/processed/pseudolabels.csv
#      data/processed/manifest.json

# 10. Train the semi-supervised transformer on gold + silver
python -m src.train_transformer \
  --train-manifest data/processed/manifest.json \
  --output models/transformer_semisup \
  --model_name /path/to/Bio_ClinicalBERT \
  --epochs 3 \
  --batch_size 8

# 11. Predict with the semi-supervised model
python -m src.predict \
  --mode transformer \
  --model models/transformer_semisup \
  --input data/raw/test.csv \
  --output outputs/test-pred-semisup.csv
```

### Scripts reference

#### `scripts/pseudolabel_mimic.py`

**Full signature:**

```bash
python scripts/pseudolabel_mimic.py \
  --mimic-csv PATH \
  [--baseline-path PATH] \
  [--output-dir PATH] \
  [--confidence FLOAT] \
  [--max-words INT] \
  [--batch-size INT] \
  [--categories "Comma,separated,list"] \
  [--sample N] \
  [--gold-csv PATH]
```

**Important details:**
- `row_id` generation: global monotonic integer counter across all processed notes (compatible with `src/contracts.py` integer types)
- Paths in `manifest.json` are stored **relative** to the manifest file's directory for portability between Mac/Linux/Grace
- Binary label enforcement: `BaselineModel` returns class indices directly; no rounding or threshold gymnastics needed beyond `predict()`
- Sentence splitting uses a simple regex (`[.!?]+`) to avoid external NLTK dependencies; sufficient for clinical prose

#### `scripts/run_pseudolabel_slurm.sh`

**Full script (annotated):**

```bash
#!/bin/bash
#SBATCH --job-name=pseudolabel-mimic
#SBATCH --output=logs/pseudolabel_%j.out
#SBATCH --error=logs/pseudolabel_%j.err
#SBATCH --time=02:00:00          # adjust upward for full MIMIC (~2h)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=normal       # adjust to your allocation (normal, short, etc.)

set -euo pipefail

# ---- User configuration (EDIT THESE) ----
PROJECT_ROOT="/home/kyler/projects/csce421-final-project"
MIMIC_CSV="/path/to/NOTEEVENTS.csv.gz"   # <── MUST SET
BASELINE_MODEL="${PROJECT_ROOT}/models/baseline_model.pkl"
OUTPUT_DIR="${PROJECT_ROOT}/data/processed"
CONFIDENCE=0.95
BATCH_SIZE=512
# ----------------------------------------

module load python/3.11      # or whatever module Grace provides
cd "${PROJECT_ROOT}"

if [ ! -d ".venv" ]; then
    python -m venv .venv
fi
source .venv/bin/activate

pip install -q -r requirements.txt
mkdir -p logs

echo "Starting pseudolabeling at $(date)"
python scripts/pseudolabel_mimic.py \
  --mimic-csv "${MIMIC_CSV}" \
  --baseline-path "${BASELINE_MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --confidence "${CONFIDENCE}" \
  --batch-size "${BATCH_SIZE}" \
  --categories "Discharge summary"
echo "Completed at $(date)"
```

### Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `manifest.json` not found by train script | `--train-manifest` points to wrong path or manifest `version` mismatch | Ensure you pass the manifest file created by the pseudolabeler; version is `1.0` and supported by `src/data.py` |
| Low silver yield (few rows) | Confidence too high for your data distribution | Lower `--confidence` to 0.90 and re-run; inspect `pseudolabels.csv` to confirm label spread |
| OOM on Grace | Batch size too big for 32 GB | Reduce `--batch-size` (try 128 or 64) in the Slurm script |
| `NOTEEVENTS` column errors | MIMIC schema differs (older/forked version) | Verify column names: `row_id,subject_id,chartdate, … ,category,text` is standard; script uses `CATEGORY` and `TEXT` (case-insensitive lookup) |
| Chunking produces empty fragments | Note has no sentence punctuation | The regex fallback will take the whole note; if still empty, the fragment is skipped (logged at `DEBUG`) |

---

## Notes

- Default transformer max length is `128`.
- The transformer wrapper is import-safe even if `torch` / `transformers` are not installed.
- Submission output is the course format: `row_id,prediction`.
- Use `--debug-output` if you also want a richer local inspection CSV.
- The transformer path is intentionally minimal; it is set up to be extended once you decide on evaluation, cross-validation, and checkpoint strategy.
- `src/contracts.py` enforces strict binary labels (0/1) across baseline and transformer predict paths — the pseudolabeler inherits this contract.

## Tests

```bash
pytest -q tests/test_smoke.py
```
