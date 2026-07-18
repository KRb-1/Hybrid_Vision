# Reconstruction Baseline Evaluation Metrics

## Evaluation setting

The reconstructed frames from E2VID and FireNet were evaluated using the Team A event-domain YOLO detector:

- Model: `package_for_teammate_visual_demo/models/event_official_best.pt`
- Class: `drone`
- Evaluation command: `yolo detect val`
- Ground truth: matched from `annotations_test_official_0_180.csv` using `alignment_test_official_0_180.csv`

## Validation result paths on server

- E2VID validation result:
  `runs/detect/memberB_eval/val_results/e2vid_val/`

- FireNet validation result:
  `runs/detect/memberB_eval/val_results/firenet_val/`

## Detection metrics

| Method | Images | GT Instances | Precision | Recall | AP50 / mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|---:|
| E2VID | 1602 | 568 | 0.000438 | 0.00528 | 0.00000421 | 0.00000171 |
| FireNet | 8790 | 2806 | 0.221 | 0.0797 | 0.0363 | 0.00846 |

## Detection coverage

| Method | Total reconstructed images | Frames with YOLO detection txt | Detection coverage |
|---|---:|---:|---:|
| E2VID | 1759 | 64 | 3.64% |
| FireNet | 10321 | 232 | 2.25% |

## Notes

Detection coverage is not equal to precision, recall, or AP50. It only measures the percentage of reconstructed frames where YOLO produced at least one detection output.

The YOLO validation metrics are the proper detection evaluation metrics because they compare predictions with ground-truth bounding boxes.
