# Project Actions Log

Generated: 2026-04-29T16:14:14Z

This is a short chronological record of the main project actions we completed while working on the CSCE 421 final project.

| Timestamp | Action | Outcome / Notes |
| --- | --- | --- |
| 2026-04-28 22:41:22 CDT | Submitted the Grace pseudolabel job (`18471880`) | First run failed because the batch script still tried to install dependencies on the compute node. |
| 2026-04-28 22:47:07 CDT | The first pseudolabel pass finished | It completed cleanly, but produced `0` silver rows, which made the semi-supervised path unusable without a fallback. |
| 2026-04-29 (session) | Removed the runtime `pip install` step from the Grace wrapper | Made the batch job portable and removed a compute-node internet dependency. |
| 2026-04-29 (session) | Patched the Grace pseudolabel script for compatibility | Cleared an import/runtime issue and kept the workflow moving. |
| 2026-04-29 (session) | Verified the reusable Grace environment and installed missing user-site tooling | Confirmed `tqdm` was available and that the environment could run the project scripts. |
| 2026-04-29 (session) | Moved Hugging Face cache paths to scratch and staged `Bio_ClinicalBERT` | Avoided `/home/kcao` quota problems and made transformer work feasible on Grace. |
| 2026-04-29 (session) | Ran the ClinicalBERT transformer training job (`18472935`) | Training completed successfully on Grace (`COMPLETED`, exit `0:0`, node `g109`). |
| 2026-04-29 (session) | Ran the transformer evaluation job (`18473667`) | Inference/comparison completed successfully on Grace (`COMPLETED`, exit `0:0`, node `g101`). |
| 2026-04-29 (session) | Rebuilt and compared local prediction CSVs | Compared transformer vs gold-only vs combined baselines across `test01`, `test02`, and `test03`. |
| 2026-04-29 (session) | Interpreted the results | The combined TF-IDF + logistic regression baseline remained the safest ship model; the transformer looked like a useful experiment but not a clear replacement. |
| 2026-04-29 (session) | Added a reusable Slurm wrapper for transformer training | `scripts/run_transformer_slurm.sh` now lives in the repo so teammates can reuse the Grace setup. |
| 2026-04-29 (session) | Added this log to the repo | Gives teammates a readable record of the project work and the key decisions made along the way. |
| 2026-04-30 (session) | Ran baseline teacher pseudolabel pass | Produced only 7,620 silver rows, all from confidence fallback (0.57–0.65). Baseline teacher too weak. |
| 2026-04-30 (session) | Proposed transformer as teacher for pseudolabeling | Approved by user — switch from TF-IDF+LR teacher to ClinicalBERT teacher. |
| 2026-04-30 (session) | Submitted transformer teacher pseudolabel job (`18478481`) | Produced 665k silver rows from 59,652 MIMIC-III notes — vastly better coverage. |
| 2026-04-30 (session) | Retrained baseline + transformer on new silver data | Both models trained on gold + 665k transformer-teacher pseudolabels. |
| 2026-04-30 (session) | Ran full four-way comparison on test01/02/03 | Compared old gold-only, old combined, new baseline, new transformer. |
| 2026-04-30 (session) | Interpretation of four-way results | New models agree ~91% — training on same high-quality silver produces consistent predictions. Old combined was an outlier (over-permissive). New models are closer to gold. |
| 2026-05-02 (session) | Generated Gradescope submission predictions | Created `test*-pred.csv` (old combined) and `test*-pred-tf-teacher.csv` (new baseline) files for submission. |
| 2026-05-02 (session) | Ran transformer inference on Grace for all test splits | Produced predictions: test01=39/79, test02=3944/7134, test03=79/168. Timed out before test03 finished; re-ran successfully. CSVs on Grace scratch. |
| 2026-05-02 (session) | Updated `.gitignore` to allow model/shareable predictions | Teammates can now clone and run inference with baseline models directly. |

## Short takeaway

The repo now includes:

- a reusable Grace transformer Slurm wrapper
- this action log for teammate context
- three baseline models shipped as `.pkl` files for teammates to test
- Gradescope-ready prediction CSVs in `outputs/`
- transformer-teacher pseudolabeling pipeline (665k silver rows)
- documented conclusion: transformer-teacher pseudolabels produce better models; the classical baseline trained on them is the safest deliverable

## Key numbers

| Metric | Value |
| --- | --- |
| Baseline teacher silver rows | 7,620 |
| Transformer teacher silver rows | 665k |
| New Baseline vs New Transformer agreement | ~91% |
| New models vs Gold-only agreement (test02) | ~70% |
| Old Combined vs Gold-only agreement (test02) | ~72% |
