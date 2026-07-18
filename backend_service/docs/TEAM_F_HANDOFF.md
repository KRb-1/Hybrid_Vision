# Team F handoff: Backend API + Docker

This document is for the web interface integration.

## What Team F needs from Team E

Team F needs the backend service and the API contract. The backend is already Dockerized and exposes Team A's updated RGB+Event fusion models plus the late-fusion baseline.

This standalone folder already contains the files Team F needs:

```text
backend_service/
  app.py
  backend/
  fred_two_stream/
  Dockerfile
  docker-compose.yml
  requirements.txt
  requirements-docker.txt
  models/
    early_fusion_best.pt
    two_fusion_best.pt
    late_fusion_rgb_best.pt
    late_fusion_event_best.pt
  team_b/
    results_summary/
    sample_detection_images/
  docs/
```

If the whole final GitLab repository is shared with Team F, they only need to enter `backend_service/`.

## Start the backend

From the project root:

```powershell
cd backend_service
docker compose up -d --build
```

Backend base URL:

```text
http://localhost:8000
```

Swagger/API docs:

```text
http://localhost:8000/docs
```

Stop the backend:

```powershell
docker compose down
```

## Endpoints Team F should use

```text
GET  /api/health
POST /api/detect
GET  /api/compare
```

## Health check

```text
GET http://localhost:8000/api/health
```

Expected:

```json
{
  "status": "ok",
  "weights_available": {
    "early_fusion": true,
    "two_fusion": true,
    "late_fusion": true
  },
  "team_b_online_available": false
}
```

Use this when the web interface starts, so the frontend can show whether the backend is ready.

## Detection API

```text
POST http://localhost:8000/api/detect
Content-Type: multipart/form-data
```

Form fields:

```text
method: two_fusion, early_fusion, or late_fusion
rgb_file: RGB image file
event_file: aligned Event image file
confidence: optional, default 0.25
use_tracking: false
```

Recommended default:

```text
method = two_fusion
confidence = 0.25
use_tracking = false
```

Important:

- Both `rgb_file` and `event_file` are required for all three methods.
- The event input is an event-frame image, not raw event stream data.
- The RGB/Event files must be a synchronized pair.
- Do not call `rgb`, `event`, `hybrid`, `e2vid`, or `firenet` as `/api/detect` methods.

## Detection response

Example response:

```json
{
  "status": "success",
  "request_id": "example",
  "method": "two_fusion",
  "execution_mode": "online",
  "detections": [
    {
      "frame": 0,
      "bbox_xyxy": [164.6, 21.5, 216.5, 66.4],
      "bbox_xywh": [164.6, 21.5, 51.9, 44.9],
      "class_id": 0,
      "class_name": "drone",
      "confidence": 0.564,
      "source": "two_fusion"
    }
  ],
  "tracks": null,
  "visualization_url": "/results/example/detection.jpg",
  "note": "Team A online hybrid model. Team B reconstruction baselines are exposed via /api/compare."
}
```

To display the result image in the frontend:

```text
http://localhost:8000 + visualization_url
```

For example:

```text
http://localhost:8000/results/example/detection.jpg
```

## JavaScript fetch example

```javascript
async function detectDrone({ rgbFile, eventFile, method = "two_fusion" }) {
  const formData = new FormData();
  formData.append("method", method);
  formData.append("rgb_file", rgbFile);
  formData.append("event_file", eventFile);
  formData.append("confidence", "0.25");
  formData.append("use_tracking", "false");

  const response = await fetch("http://localhost:8000/api/detect", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Backend error: ${response.status}`);
  }

  const result = await response.json();
  const visualizationUrl = result.visualization_url
    ? `http://localhost:8000${result.visualization_url}`
    : null;

  return { ...result, visualizationUrl };
}
```

## Team B comparison endpoint

```text
GET http://localhost:8000/api/compare
```

This endpoint returns precomputed/offline E2VID and FireNet reconstruction baseline statistics and sample image links.

Important:

- Team B methods are not online detection methods.
- They should be shown as comparison/report data.
- The `detection_coverage` value is not mAP, precision, or recall.

Frontend usage idea:

```text
Comparison panel:
- E2VID reconstructed frames / detected frames / coverage
- FireNet reconstructed frames / detected frames / coverage
- sample detection images
```

## Local test files

If Team F uses the same workspace, this pair can be used for a quick test:

```text
test_assets/sample_rgb.jpg
test_assets/sample_event.png
```

Expected rough result:

```text
two_fusion: one drone detection, confidence around 0.56
early_fusion: one drone detection, confidence around 0.52
late_fusion: one drone detection
```

Exact confidence may vary slightly by environment.

## Frontend acceptance checklist

Team F integration is working if:

```text
[ ] Docker backend starts with docker compose up -d --build
[ ] http://localhost:8000/api/health returns status ok
[ ] Web UI can select one RGB file and one Event file
[ ] Web UI can choose method two_fusion / early_fusion / late_fusion
[ ] Web UI sends multipart/form-data to /api/detect
[ ] Detection request returns HTTP 200
[ ] Web UI displays the returned visualization image
[ ] Web UI can show bbox/class/confidence from detections
[ ] Optional comparison panel can read /api/compare
```

## Known boundaries

- Current API supports single RGB/Event image-pair inference.
- Video/sequence playback belongs to the frontend workflow. If Team F wants batch/sequence upload, Team E and Team F need to agree on a ZIP or multi-file API extension.
- Tracking is not integrated; `use_tracking` should remain `false`.
- Team B reconstruction methods are exposed as offline comparison data, not as `/api/detect` methods.
