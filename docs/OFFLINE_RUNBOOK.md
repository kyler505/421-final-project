# Offline runbook (Gradescope + Canvas zip)

Course requirement: inference must run **without internet or external APIs**. This runbook assumes weights and code live on the same machine.

## 1. Environment

```powershell
cd path\to\421-final-project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For transformer training and prediction, install the optional stack (uncomment in `requirements.txt` or use `requirements-optional.txt` if present), then verify:

```powershell
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

## 2. Pin dependencies (before submission)

Capture exact versions used for the final run:

```powershell
pip freeze | Out-File -Encoding utf8 requirements-frozen.txt
```

Ship `requirements-frozen.txt` **or** document exact versions in your report. Keep `scikit-learn` major stable if you load pickled baselines across machines.

## 3. Train (examples)

**Baseline (single CSV or manifest):**

```powershell
python -m src.train_baseline --train data/raw/train_data-text_and_labels.csv --output models/baseline_model.pkl
```

**Transformer (local checkpoint directory):**

```powershell
$env:CSCE421_PRETRAINED_PATH = "D:\models\Bio_ClinicalBERT"
python -m src.train_transformer --train data/raw/train_data-text_and_labels.csv --output models/transformer_model --model_name $env:CSCE421_PRETRAINED_PATH
```

## 4. Evaluate gold labels (baseline CV)

```powershell
python -m src.run_eval --train data/raw/train_data-text_and_labels.csv --folds 5 --output outputs/cv_baseline.json
```

## 5. Produce Gradescope CSVs

```powershell
python -m src.predict --backend baseline --model models/baseline_model.pkl --input data/raw/test01_text_only.csv --output outputs/test01-pred.csv
python -m src.predict --backend baseline --model models/baseline_model.pkl --input data/raw/test02_text_only.csv --output outputs/test02-pred.csv
python -m src.predict --backend baseline --model models/baseline_model.pkl --input data/raw/test03_text_only.csv --output outputs/test03-pred.csv
```

Use `--backend transformer` and `--model models\transformer_model` when your primary artifact is the fine-tuned HF folder.

Optional trace for the report:

```powershell
python -m src.predict --backend baseline --model models/baseline_model.pkl --input data/raw/test01_text_only.csv --output outputs/test01-pred.csv --write-manifest outputs/test01_infer_manifest.json
```

## 6. Canvas zip contents

Include at minimum:

- All `src/`, `tests/`, `docs/`, `requirements.txt`, and any `requirements-frozen.txt`.
- **Fine-tuned weights** under `models/` (or document clearly if the course allows another layout) so a grader can run section 5 fully offline.
- LaTeX sources and report PDF per the handout.

Do **not** commit MIMIC or course CSVs to git if your course policy forbids it; they may still be included in the submission zip per instructor instructions.
