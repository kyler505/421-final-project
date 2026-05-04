# Project actions log

This is an archive of the major steps taken on the final project. It is not the main user guide.

## Milestones

| Date | Action | Outcome |
| --- | --- | --- |
| 2026-04-28 | First Grace pseudolabel run | Failed at first because the wrapper still tried to install dependencies on the compute node. |
| 2026-04-29 | Fixed Grace wrapper and cache paths | Made transformer work feasible on Grace scratch. |
| 2026-04-29 | Ran ClinicalBERT training and eval | Both jobs completed successfully. |
| 2026-04-30 | Switched to transformer-teacher pseudolabeling | Produced a much larger silver set than the baseline teacher. |
| 2026-05-03 | Finalized comparison matrix and submission choice | Tuned linear baseline became the recommended submit model. |
| 2026-05-04 | Completed linear sweep and validated transformer run | Confirmed the tuned linear baseline still wins. |

## Archive note

If you need the latest status, read `docs/results/project-status.md` instead of this log.
