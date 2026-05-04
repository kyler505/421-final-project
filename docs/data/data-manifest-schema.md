# Training Manifest Schema

Version: `1`

Use a JSON manifest when training data spans multiple CSV shards, such as gold labels plus generated silver labels.

## Schema

```json
{
  "version": 1,
  "entries": [
    {
      "path": "data/raw/train_data-text_and_labels.csv",
      "label_source": "gold",
      "split": "train"
    },
    {
      "path": "data/processed/pseudolabels_transformer_teacher.csv",
      "label_source": "silver",
      "split": "train"
    }
  ]
}
```

## Fields

- `version` — must be `1`
- `entries` — ordered list of CSV shards
- `path` — required path to a CSV file, relative to the manifest file unless absolute
- `label_source` — optional provenance tag such as `gold`, `silver`, or `external`
- `split` — optional logical split tag such as `train`, `val`, or `test`

## Loader behavior

- Paths are resolved relative to the manifest location.
- Shards are concatenated in the order listed.
- `label_source` and `split` are preserved as metadata where supported.
- The downstream loaders still expect the usual training columns (`row_id`, `text`, `label` or compatible aliases).

## Important note

`row_id_offset` is **not** part of the current manifest schema. If you need unique IDs for generated silver rows, handle that in the generator, not in the manifest contract.

## Example use

```bash
python -m src.train_baseline --train-manifest data/processed/manifest_transformer_teacher.json --output models/baseline_model_combined_tf_teacher.pkl
```
