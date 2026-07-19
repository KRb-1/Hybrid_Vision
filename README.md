# Group02 - AMI Hybrid RGB-Event Drone Detection

This repository contains the Group02 code submission for the AMI 2026 project.  The project focuses on drone detection with RGB frames and event-camera data from the FRED dataset.  It includes several RGB/Event fusion strategies, a reconstruction-based event baseline, tracking and trajectory prediction, plus a Dockerized backend and a small web demo.

## Repository Overview

| Path | Purpose |
| --- | --- |
| `train/early_fusion/` | Early-fusion YOLO pipeline. RGB and Event images are combined into a 6-channel input `[R, G, B, E_R, E_G, E_B]`. The folder contains scripts for building 6-channel TIFF training data, checking labels, training NaN-safe early-fusion models, validation, inference, and visualization. |
| `train/two_stream_lite/` | Two-stream-lite fusion pipeline. RGB and Event inputs are processed by a custom `RGBEventStem`, then fused before the YOLO backbone. This is the best local fusion method in the submitted experiments. The folder contains model configs, the custom stem implementation, training scripts, inference scripts, and visualization tools. |
| `train/late_fusion/` | Late-fusion baseline. Separate RGB and Event YOLO detectors are trained/inferred independently, then their predictions are merged with NMS-style fusion. This folder contains the code and scripts for late-fusion training, inference, fusion, evaluation, and visualization. |
| `train/results/` | Curated final artifacts for the three fusion methods. It contains the final weight files and three successful boxed sample prediction images per method. Full training run logs are not included in the repository, but the final checkpoints are provided. |
| `backend_service/` | Dockerized FastAPI backend for the demo. It exposes `/api/health`, `/api/detect`, and `/api/compare`; loads the early-fusion, two-stream-lite, and late-fusion checkpoints; creates detection visualizations; and provides Team B comparison metadata. |
| `HybridVision_Web/` | Flask frontend for the Hybrid Vision demo. It provides Home, Detection, and Comparison pages. The Detection page uploads paired RGB/Event images to the backend and displays predicted boxes and returned visualizations. |
| `Event_only_reconstruction/` | Reconstruction-based baseline from Member B. It converts raw event streams, runs E2VID and FireNet reconstruction, evaluates reconstructed intensity frames with the event-domain YOLO detector, and stores scripts, summary metrics, logs, and sample detection images. Full reconstruction outputs are too large and are documented as server-side result folders in its README. |
| `trackpre/` | Tracking and trajectory-prediction module. It converts detections into tracks, runs ByteTracker, trains/uses a Transformer-based trajectory predictor with event features, and stores tracking, prediction, pipeline, and visualization outputs. |
| `event-only_reconstruction/` | Empty compatibility placeholder kept from the source repository. The actual reconstruction baseline is in `Event_only_reconstruction/`. |

## Main Methods

| Method | Folder | Short Description |
| --- | --- | --- |
| Early fusion | `train/early_fusion/` | Packs RGB and Event data into one 6-channel input and adapts the YOLO first convolution. |
| Two-stream-lite fusion | `train/two_stream_lite/` | Uses separate RGB/Event stems and fuses features before the rest of YOLO. |
| Late fusion | `train/late_fusion/` | Runs RGB and Event detectors separately and merges boxes with NMS-style post-processing. Final checkpoints and sample prediction images are stored in `train/results/late_fusion/`. |
| Reconstruction baseline | `Event_only_reconstruction/` | Reconstructs intensity images from events with E2VID/FireNet, then evaluates them with the event-domain YOLO detector. |
| Tracking and prediction | `trackpre/` | Applies ByteTracker to detection outputs and predicts future drone trajectories with a Transformer model. |

## Quick Start: Backend Demo

The backend is the easiest entry point for testing the submitted detection models.

```powershell
cd backend_service
docker compose up -d --build
```

Open the FastAPI documentation page:

```text
http://localhost:8000/docs
```

Health check:

```text
http://localhost:8000/api/health
```

The main detection API is:

```text
POST http://localhost:8000/api/detect
```

Expected form fields:

```text
method: two_fusion, early_fusion, or late_fusion
rgb_file: RGB image
event_file: aligned Event image
confidence: optional confidence threshold, for example 0.25
use_tracking: optional, false by default
```

Recommended default method:

```text
two_fusion
```

## Quick Start: Web Frontend

Start the backend first, then run the Flask frontend:

```powershell
cd HybridVision_Web
pip install flask
python app.py
```

Open:

```text
http://localhost:8888
```

Main pages:

| URL | Page |
| --- | --- |
| `http://localhost:8888/` | Home |
| `http://localhost:8888/detection` | Upload RGB/Event images and run detection |
| `http://localhost:8888/comparison` | View reconstruction-baseline comparison data |

## Training and Inference Details

More detailed commands are documented inside the module READMEs:

| File | Contents |
| --- | --- |
| `train/early_fusion/README_early_fusion_en.md` | Build 6-channel TIFF data, train early-fusion models, run JPG inference, visualize predictions, and review current metrics. |
| `train/two_stream_lite/README_two_stream_lite_en.md` | Train/validate the two-stream-lite model, run direct RGB/Event inference, and visualize predictions. |
| `train/late_fusion/Train_Fuse_README.md` | Train separate RGB/Event YOLO models, run single-model inference, fuse predictions, evaluate mAP, and visualize boxes. |
| `backend_service/README_BACKEND.md` | Backend folder contents, Docker commands, API usage, expected health response, and test assets. |
| `HybridVision_Web/README_HybridVision_Web.md` | Frontend structure, dependencies, startup commands, pages, and backend API calls. |
| `Event_only_reconstruction/README_memberB_reconstruction.md` | Reconstruction baseline pipeline, processed sequences, submitted files, and validation metrics. |
| `trackpre/README.md` | Tracking, trajectory prediction, data formats, model structure, and pipeline commands. |

## Large Files and Data Notes

The repository includes training code, final model checkpoints, and selected sample prediction images for demonstrating the submitted work. Complete training run directories are not included, but the final fusion checkpoints are provided in `train/results/`. Raw FRED data and some full reconstruction outputs are not stored directly in this GitLab repository because of size limits; the reconstruction README documents the server-side result locations.

For Docker image submission, the backend can be rebuilt from `backend_service/Dockerfile` and `backend_service/docker-compose.yml`. If a generated Docker image or result archive is larger than the GitLab limit, it should be uploaded separately with the course-recommended large-file upload mechanism.

