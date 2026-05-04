# Primary model and offline checkpoints

## Recommended ship model

The current submission candidate is the **tuned linear baseline**.

Reason: it is the strongest completed submit candidate and is simpler to reproduce than the transformer path.

## Transformer is a comparison model

The ClinicalBERT/Bio_ClinicalBERT path remains useful for experiments and reporting, but it is not the default model to ship.

## Offline checkpoint usage

If you do use the transformer path, point it at a local checkpoint directory rather than a Hub ID.

Set:

```bash
export CSCE421_PRETRAINED_PATH=/path/to/local/Bio_ClinicalBERT
```

Then train or predict with the local directory.

## Artifact layout

Typical artifact types in this repo:

- `models/baseline_model.pkl` — gold-only baseline
- `models/baseline_model_sweepbest.pkl` — tuned linear baseline
- `models/baseline_model_combined*.pkl` — combined gold + silver variants
- `models/svm_model*.pkl` — SVM comparison artifacts
- transformer checkpoint directories — local HF-style folders for ClinicalBERT

## Usage reminder

The model choice should stay tied to the report and results page:

- current status: tuned linear baseline
- transformer: comparison only
