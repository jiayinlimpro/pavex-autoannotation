# Results

**Current results are in [`baseline_ablation/`](baseline_ablation/)**: the B0–B3 cumulative ablation on the 200-image manually annotated ground-truth subset, including the full pipeline (B3) reported in the paper.

| Path | Contents |
|---|---|
| `baseline_ablation/summary.csv` | One row of metrics per configuration (B0, B1, B2, B3, B3_rawmask) |
| `baseline_ablation/<config>/metrics.json` | Aggregate metrics, instance counts after each stage, export loss and provenance (commit, config and code hashes) |
| `baseline_ablation/<config>/per_instance.csv` | One row per TP, FP or FN instance, with IoU, Dice, `best_iou_any` and areas |
| `baseline_ablation/<config>/config_snapshot.json` | Exact configuration used for that configuration |
| `baseline_ablation/<config>/labels/` | Exported YOLOv8-seg labels for the 200 evaluation images |
| `baseline_ablation/equivalence_check.json` | Check that the cached B3 path reproduces the original pipeline on five images |
| `baseline_ablation/executed_notebook.ipynb` | The executed ablation notebook |
| `baseline_ablation/figures/` | Ground truth vs. B0 vs. B3 comparison figures |
| `baseline_ablation/fp_analysis/` | Contact sheets, tags (`fp_tags.csv`) and provenance for the B3 false-positive analysis |
| `seg_eval_summary.json` | **Superseded.** Earlier run (TP 241 / FP 131 / FN 24) from a code version in which nested suppression never fired; kept for reference only |

## Metric definitions

| Field | Meaning |
|---|---|
| `TP` / `FP` / `FN` | Instance counts after greedy one-to-one matching at mask IoU ≥ 0.5 |
| `precision` | TP / (TP + FP), micro-averaged over all instances |
| `recall` | TP / (TP + FN), micro-averaged over all instances |
| `f1` | Harmonic mean of precision and recall |
| `iou_mean`, `dice_mean` | Averaged over matched (TP) pairs only, i.e. mask quality rather than detection rate |
| `best_iou_any` | For FP rows, the highest IoU with any ground-truth mask; for FN rows, the highest IoU with any prediction |

## Provenance

- Ablation run: commit `fcec84f`, local CPU, Python 3.12, PyTorch 2.8.0. Every configuration uses the same cached GroundingDINO and SAM outputs and the same export → rasterise → evaluate path.
- Predictions are scored as exported YOLOv8-seg polygons, rasterised with the same reader as the ground truth. `B3_rawmask` scores the same instances on their raw SAM masks to measure the export loss.
