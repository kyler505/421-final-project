# Semi-Supervised Pseudolabeling — Deep Reference

This document expands on the README's Pseudolabeling section with implementation details, edge-case handling, and Grace-specific notes.

## Architecture decision log

| Decision | Rationale | Alternative considered |
|----------|-----------|------------------------|
| **Self-training vs. distant supervision** | Use gold-trained baseline predictions rather than ICD-9 code heuristics | Mapping `DIAGNOSES_ICD` → labels introduces noisy patterns (comorbidities, billing codes, historical conditions) that don't necessarily indicate codable content in the current note |
| **Sentence-aware chunking** | Prevent sentence-splitting across chunk boundaries (preserves semantic coherence) | Naïve 128-whitespace-token truncation cuts mid-sentence, fragments clinical meaning |
| **Class-balanced confidence + top-k fallback** | High precision first, but guarantee a non-empty silver set when the teacher is over-conservative | A hard 0.95 cutoff can yield zero rows on tiny/biasy teachers; fallback keeps the dataset usable without dropping quality controls |
| **Global integer row_id** | `row_id` in `src/contracts.py` is typed as `int`; string concatenation (`123_0`) would fail downstream | Use a single monotonic counter that never resets per-note; offsets in the manifest separate gold vs. silver ranges |
| **Pluggable teacher selection** | Preserve the baseline path, but allow a stronger transformer teacher fine-tuned on the gold set when the TF-IDF model is too uncertain | Hard-coding only the baseline teacher keeps the pipeline stalled when confidence never clears the cutoff |
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
  BaselineModel.predict_proba(batch) or transformer logits
        │
        ▼
  keep if confidence ≥ threshold
  otherwise, fill up to a minimum silver budget with
  class-balanced top-k examples
        │
        ▼
  pseudolabels.csv
  (row_id, text, label, confidence,
   source_row_id, chunk_idx, selection_reason)
        │
        ▼
  manifest.json
  {
    entries: [
      { path: "data/raw/train.csv", label_source: "gold_human" },
      { path: "data/processed/pseudolabels.csv", label_source: "silver_pseudolabel_conf_0.90" }
    ],
    teacher_mode: "baseline" | "transformer"
  }
        │
        ▼
  train_transformer --train-manifest manifest.json
        │
        ▼
  combined_dataset = gold ∪ silver
  fine-tune ClinicalBERT / Bio_ClinicalBERT
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

## Teacher route: baseline vs. transformer

The pseudolabeler now accepts a pluggable teacher backend:

- `--teacher-mode baseline` uses the existing TF-IDF + logistic-regression model.
- `--teacher-mode transformer` loads a fine-tuned sequence-classification checkpoint and uses its softmax/sigmoid scores.

Recommended higher-quality workflow:

```bash
# 1) Fine-tune a teacher on the gold set
python -m src.train_transformer \
  --train data/raw/train_data-text_and_labels.csv \
  --output models/transformer_teacher \
  --model_name emilyalsentzer/Bio_ClinicalBERT \
  --epochs 3 \
  --batch_size 8

# 2) Pseudolabel with the transformer teacher
python scripts/pseudolabel_mimic.py \
  --mimic-csv /path/to/NOTEEVENTS.csv.gz \
  --teacher-mode transformer \
  --teacher-path models/transformer_teacher \
  --output-dir data/processed
```

The current Grace Slurm wrapper preserves the baseline path by default and only switches to transformer inference when you set `TEACHER_MODE=transformer` and point `TEACHER_PATH` at a checkpoint.

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

### Known-good Grace runbook

The version that worked on Grace uses the prebuilt `tempdata` environment and avoids runtime installs entirely:

- direct Python: `/scratch/user/kcao/.conda/envs/tempdata/bin/python`
- Slurm stdout/stderr on scratch: `/scratch/user/kcao/csce421-final-project/logs/`
- project root on Grace: `/home/kcao/projects/csce421-final-project`
- no `pip install` inside the batch job
- set `PYTHONPATH` to the project root so `src` imports resolve
- set `PYTHONUNBUFFERED=1` so logs appear immediately
- submit with `/sw/local/bin/sbatch` when PATH is unreliable through `thinkpad-work`

The working Slurm wrapper now lives in `scripts/run_pseudolabel_slurm.sh` and is safe to copy to Grace as-is after you set `MIMIC_CSV`.

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
| `--time` | `02:00:00` (2 h) | Full `NOTEEVENTS` filtered to discharge summaries fits in this window on a compute node |
| `--mem` | `32G` | TF-IDF vectorizer and chunk buffers fit comfortably |
| `--cpus-per-task` | `8` | Batch inference benefits from multiple CPU cores |
| `--partition` | `short` | Good default for validation runs; increase only if needed |

**To estimate runtime locally:**
```bash
# Use the same Python entrypoint and a small sample
/scratch/user/kcao/.conda/envs/tempdata/bin/python scripts/pseudolabel_mimic.py --mimic-csv sample.csv --sample 1000
# Scale linearly: (total_notes / 1000) × elapsed_seconds ≈ wall-clock on Grace
```

### Inspecting Slurm output

```bash
# Job submitted: SBATCH writes JOBID
JOBID=$(/sw/local/bin/sbatch --parsable scripts/run_pseudolabel_slurm.sh)
echo "Job ID: $JOBID"

# Check status
squeue -j $JOBID -o '%.18i %.9T %.10M %.6D %.20R'
sacct -j $JOBID --format=JobID,State,ExitCode,Elapsed,NodeList -P -n

# Read the logs on scratch
less /scratch/user/kcao/csce421-final-project/logs/pseudolabel_${JOBID}.out
less /scratch/user/kcao/csce421-final-project/logs/pseudolabel_${JOBID}.err
```

Typical successful output tail:
```
Starting pseudolabeling at Tue Apr 28 22:41:22 CDT 2026
...
Pseudolabeling complete.
Next: train combined baseline with:
  python -m src.train_baseline --train-manifest /home/kcao/projects/csce421-final-project/data/processed/manifest.json --output models/baseline_model_combined.pkl
Completed at Tue Apr 28 22:47:07 CDT 2026
```

### Label balance and post-filter sanity checks

After the Slurm job finishes, run these quick checks on Grace:

```bash
# 1. Row count
wc -l data/processed/pseudolabels.csv

# 2. Label distribution
python -c "
import pandas as pd
print(pd.read_csv('data/processed/pseudolabels.csv')['label'].value_counts(normalize=True))
"

# 3. Manifest integrity
python -c "
import json, os
m = json.load(open('data/processed/manifest.json'))
print('Entries:', len(m['entries']))
print('Paths exist:', all(os.path.exists(os.path.join('data/processed', e['path'])) for e in m['entries']))
"
```

**If silver labels are all 0 or all 1:**
- Inspect gold training balance (`data/raw/train.csv` label distribution). The baseline inherits that bias.
- Consider retraining the baseline with class weights and re-running pseudolabeling.
- Prefer the built-in class-balanced fallback before lowering the threshold further.
- If you are still coverage-starved, tune `--min-silver-rows`, `--min-silver-fraction`, and `--min-per-class` instead of pushing the cutoff too low.
- The manifest now auto-includes the gold shard when it can find `data/raw/train.csv` (or the legacy `train_data-text_and_labels.csv`), so you should get a true gold+silver combined manifest by default.

### Important observed outcome

The old Grace run completed successfully but wrote **0 silver rows** because nothing exceeded the confidence cutoff. The updated script now keeps a class-balanced fallback set, so that failure mode is no longer expected. The run itself was healthy; the environment was not the issue.

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

- To regenerate pseudolabels with a different confidence or silver budget: delete `data/processed/pseudolabels.csv` and `data/processed/manifest.json`, adjust `--confidence`, `--min-silver-rows`, `--min-silver-fraction`, and `--min-per-class` in the Slurm script, and re-submit.
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
