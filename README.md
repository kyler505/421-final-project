# ICD-codable clinical text classifier

Offline semi-supervised text classification pipeline for the CSCE 421 final project.

## What the project does
- Binary classifier: ICD-codable vs not ICD-codable
- Single global model, not per-test wiring
- Baseline: TF-IDF + logistic regression
- Semi-supervised self-training on unlabeled MIMIC/public text
- Offline embedding model using a local Bio_ClinicalBERT checkpoint
- Weighted ensemble with one global threshold

## Layout
- `src/` — training, prediction, preprocessing, thresholding
- `mimiciii/` — copied MIMIC tables / notes
- `models/` — saved model bundles
- `outputs/` — generated submission CSVs

## Training flow
1. Optionally prepare unlabeled MIMIC sentences:
   ```bash
   python -m src.prepare_unlabeled --notes path/to/NOTEEVENTS.csv.gz --output data/unlabeled.csv
   ```
2. Train the full ensemble:
   ```bash
   python -m src.train      --train train_data-text_and_labels.csv      --unlabeled data/unlabeled.csv      --output models/model.joblib      --summary-output models/training_summary.json
   ```
3. Generate predictions:
   ```bash
   python -m src.predict      --model models/model.joblib      --input test01_text_only.csv      --output outputs/test01-pred.csv
   ```

## Notes
- Inputs are normalized and truncated to 128 words.
- The ensemble weights and threshold are chosen by leave-one-out validation on the 20 labeled rows.
- The embedding model runs offline using the cached local `Bio_ClinicalBERT` files.
