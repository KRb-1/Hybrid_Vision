# Hybrid Vision — FastAPI Backend & Docker

The FastAPI backend provides **three online RGB–Event fusion methods**, **two input workflows** (single aligned image pair and paired image-frame ZIP sequences), generated detection images/MP4s, and **precomputed** E2VID/FireNet comparison statistics. It is the inference API for [`../HybridVision_Web/`](../HybridVision_Web/).

> **Source of truth:** `backend/api.py`, `backend/inference.py`, `backend/model_registry.py`, `backend/video_io.py`, `backend/baseline.py`, `Dockerfile` and `docker-compose.yml`. Older handoff instructions describing image-only inference have been superseded by `/api/detect-sequence`.

## Deployment prerequisites — read first

The current public repository snapshot **does not ship `backend_service/models/` or the `test_assets/` example pair**. The `Dockerfile` explicitly does `COPY models ./models`; therefore Docker build will **fail** until you create/populate `models/`. The models also need valid weights for any selected inference method.

Obtain final checkpoints from the maintainers and use exactly these filenames, or set the corresponding configuration variables:

```text
backend_service/models/
├── early_fusion_best.pt
├── two_fusion_best.pt
├── late_fusion_rgb_best.pt
└── late_fusion_event_best.pt
```

For late fusion **both** RGB and Event weights are necessary. Model loading is lazy: even if `/api/health` returns `status: ok`, a detector can still fail to load at its first inference request if its weights are absent/incompatible.

## Quick start (backend only)

```bash
# From the repository root, after adding checkpoints as above:
cd backend_service
docker compose up -d --build
```

Then check:

- `http://localhost:8000/api/health`
- `http://localhost:8000/docs` (Swagger UI)

To inspect container logs:

```bash
docker compose logs -f api
```

To stop:

```bash
docker compose down
```

**Docker configuration:** The current Compose service is named `api`, publishes host port `8000`, mounts `./uploads` and `./outputs` for generated artifacts, sets default confidence `0.25`/image size `640`, and forces `INFERENCE_DEVICE=cpu`. The Docker image uses `python:3.11-slim` and installs **FFmpeg** for MP4 encoding. The **Flask frontend is not containerized by this Compose file**; start it separately as described in [the web guide](../HybridVision_Web/README_HybridVision_Web.md).

## Available methods

| `method` | Behavior | Weight file(s) |
| --- | --- | --- |
| `two_fusion` (default) | Six-channel RGB/Event array processed by custom RGBEventStem before the YOLO backbone. | `two_fusion_best.pt` |
| `early_fusion` | Creates temporary six-channel TIFF and runs Early Fusion YOLO. | `early_fusion_best.pt` |
| `late_fusion` | Runs RGB and Event YOLO detectors separately and merges detections using NMS-style late fusion. | `late_fusion_rgb_best.pt`, `late_fusion_event_best.pt` |

All three methods require paired RGB and rendered Event **image** inputs. Raw event streams are not accepted by the inference API.

## API reference

### `GET /api/health`

Returns `status`, `device_available`, `inference_device_setting`, `models_loaded`, `weights_available` (all three methods), `last_model_error`, and `team_b_online_available=false`. To check readiness for a method, inspect its `weights_available` entry, not `status` alone.

### `POST /api/detect` — single aligned image pair

**Content type:** `multipart/form-data`.

| Field | Type | Description |
| --- | --- | --- |
| `rgb_file` | file, required | RGB image: JPG/JPEG/PNG/BMP/WebP/TIF/TIFF. |
| `event_file` | file, required | Synchronized rendered Event-frame image in the same supported formats. |
| `method` | string | `two_fusion` (default), `early_fusion` or `late_fusion`. |
| `confidence` | float | Default `0.25`; valid `[0,1]`. |
| `use_tracking` | boolean | Must remain `false` (default). `true` returns HTTP **501**. |

**Example** (replace sample paths with your own synchronized pair):

```bash
curl -X POST 'http://localhost:8000/api/detect' \
  -F 'method=two_fusion' \
  -F 'confidence=0.25' \
  -F 'use_tracking=false' \
  -F 'rgb_file=@/path/to/rgb_000001.png' \
  -F 'event_file=@/path/to/event_000001.png'
```

Response includes `status=success`, `request_id`, `method`, `execution_mode=online`, `detections` (class/coordinates/confidence/source), `tracks=null`, and `visualization_url`, e.g. `/results/<request_id>/detection.jpg`. Add the base origin `http://localhost:8000` to that relative URL to display the image.

### `POST /api/detect-sequence` — aligned ZIP image sequences

**Content type:** `multipart/form-data`.

| Field | Type | Description |
| --- | --- | --- |
| `rgb_zip` | file, required | ZIP containing RGB image frames. |
| `event_zip` | file, required | ZIP containing Event-frame images. |
| `method` | string | Same three methods; default `two_fusion`. |
| `confidence` | float | Default `0.25`; valid `[0,1]`. |
| `fps` | integer | Output video FPS, default `30`; valid `1`–`120`. |
| `max_frames` | integer | Max paired frames processed, default `300`; valid `1`–`3000`. |

**Example:**

```bash
curl -X POST 'http://localhost:8000/api/detect-sequence' \
  -F 'method=two_fusion' \
  -F 'confidence=0.25' \
  -F 'fps=30' \
  -F 'max_frames=300' \
  -F 'rgb_zip=@/path/to/rgb_frames.zip' \
  -F 'event_zip=@/path/to/event_frames.zip'
```

The backend extracts each ZIP (including nested folders), recursively lists recognized image files, and **naturally sorts each set by filename**. It processes pairs by **list position**, up to `min(number of RGB frames, number of Event frames, max_frames)`; it does **not** verify frame synchronization from timestamps or align by shared filenames. Use numbered, matched image names within each ZIP and do not mix multiple unrelated sequences inside the archives.

For each frame pair, it runs the chosen model and creates an annotated detection JPG. It then invokes FFmpeg (`libx264`, `yuv420p`) to create three MP4s. The JSON response returns:

```json
{
  "status": "success",
  "request_id": "<generated_id>",
  "method": "two_fusion",
  "execution_mode": "online_sequence",
  "frame_count": 100,
  "rgb_frame_count": 100,
  "event_frame_count": 100,
  "fps": 30,
  "total_detections": 123,
  "rgb_video_url": "/results/<generated_id>/rgb_input.mp4",
  "event_video_url": "/results/<generated_id>/event_input.mp4",
  "output_video_url": "/results/<generated_id>/output_detection.mp4",
  "detections_per_frame": []
}
```

The numerical values shown above illustrate the **response shape only** and are **not measured project results**. In a real response `detections_per_frame` has an entry for every processed frame (`frame`, `timestamp`, `detections`). Outputs reside under `outputs/<request_id>/` and are statically served at `/results/...`. Videos are created from frame images, **not** from uploaded video files. There is **no** `/api/detect-video` endpoint.

### `GET /api/compare` and `POST /api/compare`

Returns online-method metadata under `team_a` and **precomputed** offline E2VID/FireNet metadata under `team_b`. The backend aggregates `team_b/results_summary/detection_summary.csv` to calculate sequence counts, reconstructed-frame counts, detected-frame counts and **detection coverage**. Sample images are served at `/team-b-samples/{method}/{filename}`.

**Coverage is not mAP, precision, or recall.** The Comparison webpage additionally shows a hardcoded mAP/precision/recall table; it does not calculate that table from `/api/compare`. E2VID and FireNet are **not** options for `/api/detect`.

## Backend folder guide

```text
backend_service/
├── app.py                       # Uvicorn import entrypoint
├── backend/
│   ├── api.py                   # All public API routes
│   ├── config.py                # Checkpoint paths, directories, environment defaults
│   ├── model_registry.py        # Lazy YOLO loading, method registry
│   ├── inference.py             # RGB/Event inference, late-fusion merging, CPU fallback
│   ├── fusion_input.py          # Six-channel input generation
│   ├── fusion.py                # NMS-style late fusion
│   ├── visualization.py         # Annotated detection image generation
│   ├── video_io.py              # ZIP extraction, natural frame sort, FFmpeg MP4 writer
│   ├── baseline.py              # Offline Team B CSV summary and image links
│   ├── schemas.py               # Pydantic single-image output models
│   └── tracking.py              # Unimplemented online tracking hook
├── fred_two_stream/             # Custom RGBEventStem and Ultralytics patch
├── team_b/                      # Offline E2VID / FireNet CSV + example detections
├── Dockerfile / docker-compose.yml
├── requirements.txt / requirements-docker.txt
└── docs/                       # API and integration handoff notes
```

The current tree is **missing `models/` and `test_assets/`**, although those folders are mentioned in earlier handoff documentation. `uploads/` and `outputs/` are generated locally and mapped to persistent host folders by Compose.

## Known limitations and troubleshooting

- **Build fails at `COPY models`:** add the missing `backend_service/models/` directory before rebuilding; fill it with final weights for the method(s) you want to run.
- **Health says OK but detection fails:** check `weights_available` and `last_model_error`; models load only on first use.
- **Long sequence requests:** the endpoint is synchronous, capped at `max_frames`; there is no async job queue/progress endpoint.
- **ZIP pairing problems:** matched ordering is caller responsibility; natural sorting by filename does not guarantee true time sync.
- **No raw event ingestion:** convert event data to compatible frame images first.
- **No online tracking:** `use_tracking=true` raises 501; see separate [`../trackpre/`](../trackpre/) experiments.
- **CPU-only Compose default:** CPU inference may be slow for sequences. Changing device settings or GPU deployment requires a properly configured host/container.
- **No complete Dockerized website:** the Flask frontend runs outside this Compose service.

[Main README](../README.md) · [Frontend guide](../HybridVision_Web/README_HybridVision_Web.md) · [API handoff](docs/API_HANDOFF.md)
