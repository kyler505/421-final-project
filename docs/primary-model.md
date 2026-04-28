# Primary model (recommended path)

## Offline and reproducibility

1. Download a clinical encoder checkpoint once (for example Bio_ClinicalBERT) to a path on disk **inside your machine** or course VM.
2. Point training and inference at that directory only (no Hub IDs in the final Gradescope / TA runbook).

Set `CSCE421_PRETRAINED_PATH` to that directory so [src/config.py](../src/config.py) `get_config().model_name` defaults to your local snapshot when `--model_name` is omitted.

## Artifact layout

- **Primary (expected):** Hugging Face save directory produced by `python -m src.train_transformer` containing `config.json`, tokenizer files, and PyTorch weights. Run `python -m src.predict --backend transformer --model <that_dir> ...`.
- **Baseline:** pickled `BaselineModel` for smoke tests, ablations, and sanity checks. Keep it trainable without GPU; cite it as a baseline row in the report.

Training writes `run_manifest.json` next to the checkpoint (or `--manifest`) recording `pretrained_source`, `max_length`, and hyperparameters for the Methods section.
