# Hybrid Vision — Frontend / Backend API Contract

**Verified against `backend/api.py` and the current Flask JavaScript.** API origin: `http://localhost:8000`; browser application: `http://localhost:8888`.

## Supported routes

| HTTP method | Endpoint | Function |
| --- | --- | --- |
| `GET` | `/api/health` | Server status, device, loaded/available weights. |
| `POST` | `/api/detect` | Single aligned RGB/Event image-pair detection. |
| `POST` | `/api/detect-sequence` | Paired RGB/Event ZIP image-frame sequences to detection videos. |
| `GET` / `POST` | `/api/compare` | Precomputed reconstruction-baseline stats and example images. |

**Not implemented:** `/api/detect-video`, direct MP4 upload, raw-event-stream inference, and online tracking. `use_tracking=true` on `/api/detect` gives HTTP 501.

## Model selection

`method` must be exactly `two_fusion` (default), `early_fusion`, or `late_fusion`. Event reconstruction baselines `e2vid` and `firenet` are **not online detection methods**. All three detection methods need both RGB and Event frame images.

## 1. `POST /api/detect`

`multipart/form-data` fields:

| Field | Required | Type / default |
| --- | --- | --- |
| `rgb_file` | Yes | Supported image file: JPG/JPEG/PNG/BMP/WebP/TIF/TIFF. |
| `event_file` | Yes | Matching rendered Event-frame image. |
| `method` | No | `two_fusion`. |
| `confidence` | No | float `0.25`, between `0.0` and `1.0`. |
| `use_tracking` | No | bool `false` (true is unsupported). |

Response JSON includes `status: "success"`, `request_id`, `method`, `execution_mode: "online"`, `detections`, `tracks: null`, `visualization_url`, and a note. Each detection includes `frame`, `bbox_xyxy`, `bbox_xywh`, `class_id`, `class_name`, `confidence`, and `source`. To display the annotated image, prefix the relative `visualization_url` with the backend origin, e.g. `http://localhost:8000/results/<id>/detection.jpg`.

```javascript
const form = new FormData();
form.append('rgb_file', rgbFile);
form.append('event_file', eventFile);
form.append('method', 'two_fusion');
form.append('confidence', '0.25');
form.append('use_tracking', 'false');
const response = await fetch('http://localhost:8000/api/detect', { method: 'POST', body: form });
const result = await response.json();
if (!response.ok) throw new Error(result.detail || `HTTP ${response.status}`);
const imageUrl = `http://localhost:8000${result.visualization_url}`;
```

## 2. `POST /api/detect-sequence`

`multipart/form-data` fields:

| Field | Required | Type / default |
| --- | --- | --- |
| `rgb_zip` | Yes | ZIP containing RGB frame images. |
| `event_zip` | Yes | ZIP containing Event-frame images. |
| `method` | No | `two_fusion`. |
| `confidence` | No | float `0.25`, range `0.0` to `1.0`. |
| `fps` | No | integer `30`, range 1–120; encoding/playback FPS. |
| `max_frames` | No | integer `300`, range 1–3000; cap on pairs processed. |

The backend extracts ZIPs, finds image files recursively, **natural-sorts by filename**, pairs by their **position** and processes at most the smaller number of frames or `max_frames`. Uploaders must prepare *synchronized, consistently named and ordered* RGB/Event sequences beforehand. No timestamp alignment is performed.

Returns `status`, `request_id`, `method`, `execution_mode: "online_sequence"`, `frame_count`, `rgb_frame_count`, `event_frame_count`, `fps`, `total_detections`, `detections_per_frame`, `rgb_video_url`, `event_video_url`, `output_video_url`, and `note`. A per-frame record contains `{ frame, timestamp, detections }` with timestamp computed as `frame/fps`.

```javascript
const form = new FormData();
form.append('rgb_zip', rgbZip);
form.append('event_zip', eventZip);
form.append('method', 'two_fusion');
form.append('confidence', '0.25');
form.append('fps', '30');
form.append('max_frames', '300');
const response = await fetch('http://localhost:8000/api/detect-sequence', { method: 'POST', body: form });
const result = await response.json();
if (!response.ok) throw new Error(result.detail || `HTTP ${response.status}`);
const rgbUrl = `http://localhost:8000${result.rgb_video_url}`;
const eventUrl = `http://localhost:8000${result.event_video_url}`;
const detectionUrl = `http://localhost:8000${result.output_video_url}`;
```

Three MP4s are encoded by the backend with FFmpeg, and all three can be played using `<video controls>`. Current `HybridVision_Web/static/Detection.js` synchronizes play/pause/seek/playback speed across three elements and offers Play All/Pause All/Restart controls. **It does not accept direct `.mp4` uploads.**

## 3. `GET /api/compare` (also accepts POST)

Returns a `team_a` object (online method names/default) and `team_b` object with `execution_mode="precomputed"`, `online_inference_available=false`, and `methods` aggregated from `team_b/results_summary/detection_summary.csv`. Each E2VID/FireNet entry can contain `sequences`, `reconstructed_frames`, `detected_frames`, `detection_coverage`, `sample_images`, and `metric_note`. Image URLs such as `/team-b-samples/e2vid/....jpg` are relative to the backend origin.

**Note:** `detection_coverage` means `detected_frames/reconstructed_frames`; it is **not** AP, mAP, precision or recall. The separate Precision/Recall/mAP table in `HybridVision_Web/templates/Comparison.html` is **static HTML** and not populated by this API.

## 4. Operational notes

- **Checkpoints missing in public snapshot:** `backend_service/models/` is absent from GitHub `main`; the Dockerfile requires it via `COPY models ./models`. Four checkpoint filenames are documented in [`../README_BACKEND.md`](../README_BACKEND.md). Missing weights can cause first inference to fail even if health reports OK.
- **Sample test files missing:** Older notes referencing `test_assets/sample_rgb.jpg` and `sample_event.png` are outdated; use your own matching files.
- **Docker/ports:** Compose publishes `8000`, defaults to CPU inference, and encodes MP4s with the bundled FFmpeg. Flask on `8888` starts separately.
- **CORS:** Current backend permits cross-origin browser requests; frontend currently hardcodes API origin `http://localhost:8000`.
- **Output media:** `/results/<id>/...` comes from backend `outputs`; `/team-b-samples/...` exposes available offline sample images.

[Backend guide](../README_BACKEND.md) · [Web guide](../../HybridVision_Web/README_HybridVision_Web.md)
