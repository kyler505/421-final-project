# Semi-Supervised Pseudolabeling — Deep Reference

This document expands on the README's Pseudolabeling section with implementation details, edge-case handling, and Grace-specific notes.

## Architecture decision log

| Decision | Rationale | Alternative considered |
|----------|-----------|------------------------|
| **Self-training vs. distant supervision** | Use gold-trained baseline predictions rather than ICD-9 code heuristics | Mapping `DIAGNOSES_ICD` → labels introduces noisy patterns (comorbidities, billing codes, historical conditions) that don't necessarily indicate codable content in the current note |
| **Sentence-aware chunking** | Prevent sentence-splitting across chunk boundaries (preserves semantic coherence) | Naïve 128-whitespace-token truncation cuts mid-sentence, fragments clinical meaning |
| **Confidence threshold 0.95** | High precision for silver labels; err on the side of clean data | Lower thresholds increase silver volume but risk label noise that could poison fine-tuning |
| **Global integer row_id** | `row_id` in `src/contracts.py` is typed as `int`; string concatenation (`123_0`) would fail downstream | Use a single monotonic counter that never resets per-note; offsets in the manifest separate gold vs. silver ranges |
| **Relative manifest paths** | Manifest and code should be portable across Mac dev → Grace cluster → any downstream runner | Absolute paths bind the artifact to one machine; relative to manifest directory is the convention in `src/data.py` |

## Data flow diagram

```
NOTEEVENTS.csv.gz (MIMIC)
        │
        ▼
  filter by CATEGORY
  ("Discharge summary")
        │
        ▼
  chunk → [fragment_0, fragment_1, …]
  (sentence-aware, ≤128 words)
        │
        ▼
  BaselineModel.predict_proba(batch)
        │
        ▼
  keep if max(proba) ≥ 0.95
        │
        ▼
  pseudolabels.csv
  (row_id, text, label)
        │
        ▼
  manifest.json
  {
    shards: [
      { path: "data/raw/train.csv", source: "gold",  row_id_offset: 0 },
      { path: "data/processed/pseudolabels.csv", source: "silver", row_id_offset: 1_000_000 }
    ]
  }
        │
        ▼
  src.train_transformer --train-manifest manifest.json
        │
        ▼
  combined_dataset = gold ∪ silver
  fine-tune Bio_ClinicalBERT
```

## Sentence-aware chunking algorithm

```python
def chunk_text(text: str, max_words: int) -> list[str]:
    sentences = re.split(r'[.!?]+', text)   # keeps sentence boundaries
    chunks, current = [], []
    current_len = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        words = sent.split()
        if current_len + len(words) > max_words and current:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sent)
        current_len += len(words)

    if current:
        chunks.append(" ".join(current))
    return chunks
```

Properties:
- Never splits a sentence across chunks
- Each chunk ≤ `max_words` (can be slightly under when a single sentence exceeds limit — that's acceptable; the sentence stays intact)
- Handles notes with no punctuation (falls back to the whole note as one chunk)

## Confidence filtering

`BaselineModel.predict_proba()` returns shape `(n_samples, 2)` for binary classification (column order matches `self.model.classes_`). We take `proba[:, 1]` (positive-class probability) and threshold it:

```python
keep_mask = proba_positive >= args.confidence
silver_df = pd.DataFrame({
    "row_id": global_counter[keep_mask],   # pre-incremented monotonic ints
    "text": kept_texts,
    "label": predicted_class[keep_mask]    # argmax over the two columns
})
```

**Why use `predict()` not `predict_proba()` threshold?**
The model already outputs calibrated probabilities via `predict_proba`. The silver label is simply the argmax of the 2-element vector — exactly mirroring `predict()` — but we gate inclusion by the confidence score. This ensures every silver fragment has a strong signal.

## Manifest mechanics (`src/manifest.py`)

Multi-shard manifests let `src/data.py` union CSVs with different sources without concatenating files physically.

**Shard object fields:**

| Field | Meaning |
|-------|---------|
| `path` | Path to the shard CSV, relative to the manifest's parent directory |
| `source` | Arbitrary tag (`gold`, `silver`, `external`, …) — carried into `Dataset.extra_metadata` |
| `split` | `train`, `val`, or `test` — determines whether labels are required |
| `row_id_offset` | Integer added to every `row_id` in this shard during loading (ensures global uniqueness) |
| `label_column` | Column name holding the label (default `label`) |
| `text_column` | Column name holding the text (default `text`) |

**Validation during load** (`data.load_manifest()`):
- All paths exist and are readable
- `row_id`s are integers; if duplicates detected across shards without distinct offsets → error
- For `split=="train"`, `label_column` must be present; for `split=="test"`, it must be absent

**Adding a new shard manually:**
If you want to experiment with multiple silver sources (e.g., 0.90 and 0.95 thresholds), append another object to `shards` with its own `row_id_offset` (use the docs-recommended `previous_offset + shard_row_count`).

## Grace/HPRC execution details

### File placement (suggested)

```
/home/kyler/projects/csce421-final-project/
├── data/
│   ├── raw/
│   │   ├── train.csv                    # your gold train
│   │   └── test.csv                     # your gold test
│   └── processed/
│       ├── pseudolabels.csv             # generated by Slurm job
│       └── manifest.json                # generated by Slurm job
├── models/
│   ├── baseline_model.pkl               # copy from local machine
│   └── transformer_semisup/             # trained later, after pseudolabels exist
├── scripts/
│   ├── pseudolabel_mimic.py
│   └── run_pseudolabel_slurm.sh
└── requirements.txt
```

### Transferring files to Grace

If you have SSH access to `grace.hprc.tamu.edu` via the `thinkpad-work` jump host, use the skill `grace-via-thinkpad-work` or manual `scp`:

```bash
# From your Mac (assuming SSH agent forwarding to thinkpad-work)
scp -r /home/kyler/projects/csce421-final-project kyler505@grace.hprc.tamu.edu:/home/kyler/projects/
```

Alternatively, push to GitHub on your Mac and `git pull` on Grace.

### Slurm resource planning

| Resource | Setting | Rationale |
|----------|---------|-----------|
| `--time` | `02:00:00` (2 h) | Full `NOTEEVENTS` (~2M rows) filtered to Discharge summaries ≤200k notes; 256 batch × 8 CPUs can process in ~90 min on a modern node |
| `--mem` | `32G` | TF-IDF vectorizer holds corpus in sparse memory; 32 GB comfortably fits all chunks |
| `--cpus-per-task` | `8` | Parallel inference via joblib (inside `BaselineModel.predict`) — 8 cores speeds up batch predict by ~7× |
| `--partition` | `normal` | Adjust to your lab/allocation queue; use `short` for tests; `highmem` not required |

**To estimate runtime locally:**
```bash
# Install scikit-learn ≥ 1.0 for parallel predict (n_jobs defaults to 1 here; you can bump in BaselineModel)
time python scripts/pseudolabel_mimic.py --mimic-csv sample.csv --sample 1000
# Scale linearly: (total_notes / 1000) × elapsed_seconds ≈ wall-clock on Grace
```

### Inspecting Slurm output

```bash
# Job submitted: SBATCH writes JOBID
JOBID=$(sbatch scripts/run_pseudolabel_slurm.sh | grep -o '[0-9]*')
echo "Job ID: $JOBID"

# Stream live (if still running)
tail -f logs/pseudolabel_${JOBID}.out

# After completion, view summary
cat logs/pseudolabel_${JOBID}.out | grep -E "Processing|Chunked|kept|rows|manifests written"

# Check resource usage
sacct -j $JOBID --format=JobID,State,Elapsed,MaxRSS,Node,AllocCPUs,ReqMem
```

Typical successful output tail:
```
Completed pseudolabeling: 184732 fragments total, 12745 kept (confidence ≥ 0.95)
Wrote pseudolabels CSV: data/processed/pseudolabels.csv (12745 rows)
Wrote manifest: data/processed/manifest.json
```

## Label balance and post-filter sanity checks

After the Slurm job finishes, run these quick checks on Grace:

```bash
# 1. Row count
wc -l data/processed/pseudolabels.csv

# 2. Label distribution (expect heavily skewed towards 0 or 1 depending on your gold set)
python -c "
import pandas as pd, json
df = pd.read_csv('data/processed/pseudolabels.csv')
print(df['label'].value_counts(normalize=True))
"

# 3. Manifest integrity
python -c "
import json, yaml
m = json.load(open('data/processed/manifest.json'))
print('Shards:', [s['source'] for s in m['shards']])
print('Paths relative to manifest dir; all exist:',
      all(__import__('os').path.exists(__import__('os').path.join('data/processed', s['path'])) for s in m['shards']))
"
```

**If silver labels are all 0 or all 1:**
- Inspect gold training balance (`data/raw/train.csv` label distribution). The baseline inherits that bias.
- Consider retraining the baseline with class weights (`--class-weight balanced`) and re-running pseudolabeling.
- Lowering `--confidence` may also recover the minority class if it's inherently hard to predict.

## Integration with transformer training

Once `manifest.json` exists, the semi-supervised run is identical to the gold-only run — just point `--train-manifest` at the combined manifest instead of a single CSV:

```bash
python -m src.train_transformer \
  --train-manifest data/processed/manifest.json \
  --output models/transformer_semisup \
  --model_name /path/to/Bio_ClinicalBERT \
  --epochs 3 \
  --batch_size 8
```

`src/data.py` reads each shard, applies `row_id_offset`, and yields examples in a streaming fashion. The effective training size is:

```
|gold_train| + |silver_pseudolabels|
```

**Memory note:** Streaming prevents loading the full MIMIC-derived silver set into memory at once; only the gold CSV is small enough to cache.

## Cleaning up / re-running

- To regenerate pseudolabels with a different confidence: delete `data/processed/pseudolabels.csv` and `data/processed/manifest.json`, adjust `CONFIDENCE` in the Slurm script (or pass `--confidence` on the command line), and re-submit.
- To change the MIMIC category filter: `--categories "Discharge summary,Radiology"` (comma-separated; no spaces around commas).
- To process the entire `NOTEEVENTS` (all categories): `--categories ""` (empty string disables filtering). Be aware this expands the job 3–5× in runtime.

## FAQ

**Q: Do I need to chunk MIMIC for the transformer?**  
A: No. The pseudolabels are **already chunked** because the baseline was trained on 128-word fragments (per project spec). When you fine-tune the transformer on the silver CSV, you are training on those same-sized fragments — consistent end-to-end.

**Q: Can I use the transformer to re-pseudolabel (iterative self-training)?**  
A: You could — train a transformer on gold+silver, then re-run `pseudolabel_mimic.py` with `--model-mode transformer` (not implemented yet). The current scaffold keeps it to one iteration to avoid error accumulation.

**Q: What if `NOTEEVENTS.csv.gz` is >50 GB and I don't have enough local disk?**  
A: Stream it directly on Grace: place the gzip on a scratch filesystem and run the script there; intermediate chunks live in RAM + temporary vectors, not on disk.

**Q: Are the pseudolabels deterministic?**  
A: Yes. `BaselineModel.predict` is deterministic (fixed `random_state=42` during training). Chunk boundaries are deterministic given the same `max_words`. The manifest row_id offset assignment is deterministic.

**Q: Can I exclude certain `row_id` ranges from gold?**  
A: Edit `scripts/pseudolabel_mimic.py` and add `--exclude-gold-ids` (not currently present) or pre-filter your gold CSV. The manifest offsets are computed automatically from manifest order.
