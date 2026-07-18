# Member B Reconstruction Baseline

## Task

This part implements the reconstruction-based baseline for the AMI Hybrid Vision project.

The goal is to compare detection performance on intensity images generated from event-based video reconstruction methods.

Two reconstruction methods were used:

- E2VID
- FireNet

The reconstructed intensity images were then evaluated using the Team A event-domain YOLO detector.

## Pipeline

The workflow is:

raw event stream
→ convert raw events into events_txyp.npy and events_firenet.txt
→ E2VID reconstruction from events_txyp.npy
→ FireNet reconstruction from events_firenet.txt
→ reconstructed intensity images
→ Team A event-domain YOLO prediction
→ YOLO detection images and bbox txt files
→ frame-level detection coverage
→ match reconstructed frames with ground-truth annotations using alignment files
→ build YOLO validation datasets
→ YOLO validation metrics: precision, recall, AP50, and mAP@0.5:0.95

## Implemented work

This part includes:

- Raw event conversion into `events_txyp.npy`
- FireNet input generation as `events_firenet.txt`
- E2VID reconstruction
- FireNet reconstruction
- YOLO prediction on E2VID reconstructed frames
- YOLO prediction on FireNet reconstructed frames
- Detection output saving, including visual prediction images and bbox txt files
- Frame-level detection coverage calculation
- Reconstruction frame alignment with ground-truth annotations
- YOLO validation dataset generation
- YOLO validation for precision, recall, AP50, and mAP@0.5:0.95
- Result summary, logs, scripts, and sample detection images

## Processed sequences

8, 9, 12, 20, 21, 29, 34, 39, 41, 46,
51, 60, 63, 64, 65, 80, 83, 90, 93,
110, 111, 114, 115, 127, 128, 129,
138, 139, 142, 152, 153, 154,
162, 163, 167, 178, 179

## Full result folders on server

The full result folders are too large for GitLab, so they are kept on the server.

Reconstruction outputs:

- memberB_results/reconstruction/e2vid/
- memberB_results/reconstruction/firenet/

YOLO detection outputs:

- memberB_results/yolo_detection/e2vid/
- memberB_results/yolo_detection/firenet/

Converted event files:

- memberB_results/converted_events/

YOLO validation datasets:

- memberB_eval/e2vid_dataset/
- memberB_eval/firenet_dataset/

YOLO validation result paths:

- E2VID: runs/detect/memberB_eval/val_results/e2vid_val/
- FireNet: runs/detect/memberB_eval/val_results/firenet_val/

## Submitted files

This submission folder contains:

- scripts/raw_to_npy_safe.py
- scripts/make_firenet_txt_stream.py
- results_summary/detection_summary.csv
- results_summary/reconstruction_eval_metrics.md
- logs/
- sample_detection_images/

## Detection coverage

Detection coverage means:

number of frames with YOLO bbox txt / total reconstructed frames

This is not the same as precision, recall, or AP50.

| Method | Total reconstructed images | Frames with YOLO detection txt | Detection coverage |
|---|---:|---:|---:|
| E2VID | 1759 | 64 | 3.64% |
| FireNet | 10321 | 232 | 2.25% |

## Validation metrics

YOLO validation was performed for the reconstruction-based baseline using ground-truth annotations.

Evaluation setting:

- Model: package_for_teammate_visual_demo/models/event_official_best.pt
- Class: drone
- Ground truth: matched from annotations_test_official_0_180.csv using alignment_test_official_0_180.csv
- Evaluation command: yolo detect val

Main results:

| Method | Images | GT Instances | Precision | Recall | AP50 / mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|---:|
| E2VID | 1602 | 568 | 0.000438 | 0.00528 | 0.00000421 | 0.00000171 |
| FireNet | 8790 | 2806 | 0.221 | 0.0797 | 0.0363 | 0.00846 |

## Notes

The reconstruction baseline is not a separate trained detection model.

This part uses E2VID and FireNet to reconstruct intensity images from raw event streams, and then evaluates these reconstructed images using the Team A event-domain YOLO detector.

The full reconstruction images and YOLO detection outputs are not included in this submission folder because they are large. They are stored on the server paths listed above.