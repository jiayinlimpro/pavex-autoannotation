# Results

Evaluation on the 200-image manually annotated ground-truth subset, as reported in Table 4 of the paper.

| File | Contents |
|---|---|
| `seg_eval_summary.json` | Aggregate metrics over the 200 ground-truth images |

## Metric definitions

| Field | Meaning |
|---|---|
| `TP` / `FP` / `FN` | Instance counts after greedy one-to-one matching at mask IoU ≥ 0.5 |
| `precision` | TP / (TP + FP), micro-averaged over all instances |
| `recall` | TP / (TP + FN), micro-averaged over all instances |
| `IoU_mean`, `Dice_mean` | Averaged over matched (TP) instances only, i.e. mask quality rather than detection rate |
| `neg_imgs` | Ground-truth images with no potholes (none in this subset) |
| `FP_per_neg_img` | False positives per negative image; `null` because there are no negative images |
| `skipped_labels` | Ground-truth labels whose image could not be found (none) |

## Provenance

- Produced on Google Colab (GPU) in November 2025 with the configuration in `configs/`.
- Re-running the pipeline may give slightly different instance counts, due to GPU non-determinism and library-version differences. A local re-run of the current notebook on a 50-image subset agreed closely with earlier runs.
- New runs write their own `seg_eval_summary.json` and `seg_eval_per_image.csv` to `runs/run_<timestamp>/eval/`.
