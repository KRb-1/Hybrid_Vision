# Hybrid Vision — Flask Web Frontend

This folder provides the **browser interface** for the TUM AMI Group 02 RGB–Event Drone Detection project. Flask serves the HTML pages on port **8888**; the browser's JavaScript directly calls a separate FastAPI backend at `http://localhost:8000`.

> **Verified against current code:** `app.py`, `templates/*.html`, `static/Detection.js`, and `static/Comparison.js`. This is **not** a standalone detector: you need the backend and relevant YOLO weights for live detection.

## Features and routes

| URL | Page | What users can do |
| --- | --- | --- |
| `/` | Home | See the project introduction, workflow and detection entry point. |
| `/detection` | Detection | Run single RGB/Event image-pair detection **or** ZIP frame-sequence detection; choose among three methods; display image/video results. |
| `/comparison` | Comparison | Read E2VID/FireNet offline coverage statistics and sample images from the API; see a separate, **static HTML table** of reported detection metrics. |
| `/tracking` | Tracking | Read about the ByteTrack → Transformer pipeline; view/download **precomputed** media when the expected static files are present. |

## Detection page: exact implemented behavior

The Input Type selector in `templates/Detection.html` offers **two**, not three, options:

### 1. `Single image pair`

1. Upload an aligned **RGB Image** and **Event Image** (rendered Event frame, not raw events).
2. Choose `two_fusion` (default), `early_fusion`, or `late_fusion`.
3. Set confidence between 0 and 1 (default `0.25`) and select **Run Detection**.
4. The browser calls `POST /api/detect` with `rgb_file`, `event_file`, `method`, `confidence`, and `use_tracking=false`.
5. The page shows the input previews, annotated RGB detection image, status, method, number of detections and a table of class, confidence, source, `bbox_xyxy` and `bbox_xywh`.

### 2. `Frame sequence zip`

1. Upload **RGB Frames ZIP** and **Event Frames ZIP**, containing already prepared frame images.
2. Choose a model and optionally change **Playback FPS** (default `30`; 1–120) and **Max Frames** (default `300`; 1–3000).
3. Select **Run Sequence Detection**. The browser calls `POST /api/detect-sequence` using `rgb_zip`, `event_zip`, `method`, `confidence`, `fps`, `max_frames`.
4. The server processes the aligned image pairs and returns `rgb_video_url`, `event_video_url`, `output_video_url`, `frame_count`, `total_detections`, and `detections_per_frame`.
5. The site displays the resulting **RGB Input Video**, **Event Input Video** and **Detection Output Video** side by side (video layout may wrap by screen width). It offers **Play All**, **Pause All**, **Restart**, and speed options **0.5×/1×/1.5×/2×**. Playback events, seeking, rates, and time updates are synchronized by `static/Detection.js`.

**Prepare inputs carefully:** ZIP files are recursively collected and naturally sorted by filename; corresponding positions are treated as synchronized pairs. Files are not paired by timestamp. Use matching lists such as:

```text
rgb_frames.zip                  event_frames.zip
└── 000001.png                  └── 000001.png
└── 000002.png                  └── 000002.png
└── 000003.png                  └── 000003.png
```

Supported image suffixes include `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`, and `.tiff`. These are image-frame ZIPs; **direct MP4 video upload, raw event streams, and a `/api/detect-video` endpoint are not implemented**.

## Comparison page

`static/Comparison.js` calls `GET /api/compare` and renders:

- A list of online fusion methods reported by the backend.
- Cards for **E2VID** and **FireNet** containing number of sequences, reconstructed frames, frames with detection outputs and detection coverage.
- Sample detection-image galleries supplied by the API; a Refresh button reloads the data.

**Metric caveat:** The Precision / Recall / mAP table near the top of `templates/Comparison.html` is hardcoded HTML (it is *not* filled from the API). The API's `detection_coverage` is the fraction of reconstructed images producing detection files, **not mAP or recall**. The reconstruction methods remain offline baselines and are not options in the Detection model selector.

## Tracking page

`templates/Tracking.html` is a **static informational showcase** of ByteTrack association and Transformer trajectory forecasting (12-frame observations and 24-frame future predictions are documented in [`../trackpre/README.md`](../trackpre/README.md)). The HTML expects these assets:

```text
HybridVision_Web/static/tracking/seq51/
├── output.mp4
├── tracks.txt
└── predictions_future.txt
```

**These files are not currently committed to the public repository.** The page route exists, but playback/download links will not work unless the assets are added at the expected locations. This page does **not** call a live tracking endpoint or execute the `trackpre/` pipeline.

## Frontend directory guide

```text
HybridVision_Web/
├── app.py                         # Flask routing: /, /detection, /comparison, /tracking
├── templates/
│   ├── Home.html                  # Project overview / entry page
│   ├── Detection.html             # Two detection modes and video controls
│   ├── Comparison.html            # Static metrics table + API-loaded baseline panels
│   └── Tracking.html              # Static tracking showcase
├── static/
│   ├── Detection.js               # Health check, form submission, result rendering, video sync
│   ├── Comparison.js              # Read /api/compare and populate offline baseline cards
│   ├── Home.css / Detection.css / Comparison.css / Tracking.css
│   ├── jquery.js / nicepage.js / nicepage.css
│   └── images/                    # Site visual assets
└── models/detector.py             # Legacy placeholder: NOT called by current Flask app
```

The old `models/detector.py` copies an input file rather than running the real detector. It is retained in the repository but **not used** by the current Flask app; actual inference is performed by FastAPI.

## Start the application

**Prerequisites:** Python 3, `flask`, running FastAPI backend on `localhost:8000`, and backend model weights. The current backend Dockerfile expects checkpoint files under `backend_service/models/` (not shipped in the public `main` branch).

**Terminal 1 — Backend:** from the repository root, after supplying weights:

```bash
cd backend_service
docker compose up -d --build
```

Check `http://localhost:8000/api/health` for both `status` and `weights_available`. Swagger UI: `http://localhost:8000/docs`. See [backend setup](../backend_service/README_BACKEND.md) for current missing-artifact caveats.

**Terminal 2 — Frontend:** from the repository root:

```bash
cd HybridVision_Web
python -m pip install flask
python app.py
```

Open **http://localhost:8888**. Visit `/detection`, `/comparison`, or `/tracking` as needed. Stop the Flask process with `Ctrl+C`; stop the backend with `docker compose down` in `backend_service/`.

## Backend routes used

| Endpoint | Used by | Response |
| --- | --- | --- |
| `GET /api/health` | Detection | Availability/status information. |
| `POST /api/detect` | Detection image mode | Detection JSON and image `visualization_url`. |
| `POST /api/detect-sequence` | Detection ZIP mode | Summary, per-frame detections and three generated MP4 URLs. |
| `GET /api/compare` | Comparison | Precomputed reconstruction statistics and sample URLs. |

The backend additionally allows `POST /api/compare`, but the frontend uses GET. `use_tracking=true` is not implemented by the backend and gives HTTP 501; the frontend sets it to `false` for image detection.
