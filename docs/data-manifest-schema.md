# Training manifest schema (version 1)

Use a JSON manifest when training data spans multiple CSV shards (for example, course gold labels plus local MIMIC-derived silver labels under `data/processed/`). Paths are resolved relative to the manifest file’s directory unless absolute.

## Fields

- `version` (int): must be `1`.
- `entries` (array): each element is an object with:
  - `path` (string, required): CSV with columns compatible with `load_train_data` (`row_id`, `text`, `label` or inferable aliases).
  - `label_source` (string, optional): provenance tag, e.g. `gold`, `silver_heuristic`. Defaults to `unknown`.
  - `split` (string, optional): logical split tag for reporting, e.g. `gold_train`, `silver_pool`. Defaults to `none`.

The loader concatenates all shards and adds `label_source` and `split` columns. Downstream code that only needs `text` and `label` can keep using `get_texts_labels`.

## Example

See [training_manifest.example.json](examples/training_manifest.example.json).
