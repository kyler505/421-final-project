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

## Short takeaway

The repo now includes:

- a reusable Grace transformer Slurm wrapper
- this action log for teammate context
- the existing pseudolabeling documentation and runbook material

The main technical conclusion from the work is still the same: the combined classical baseline is the safest final model, and ClinicalBERT is useful as a comparison experiment rather than a drop-in replacement.
