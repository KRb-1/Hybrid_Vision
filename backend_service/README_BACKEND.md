# Backend service for Team F

This folder is the Dockerized backend service for the AMI Hybrid Vision project.

It exposes Team A's updated RGB+Event fusion models and the late-fusion baseline:

- `two_fusion` - default, two-stream-lite feature-level fusion
- `early_fusion` - 6-channel input-level fusion
- `late_fusion` - RGB detector + Event detector merged with NMS

It also exposes Team B's E2VID/FireNet reconstruction baseline as offline comparison metadata through `/api/compare`.

## Folder contents

```text
backend_service/
  app.py
  backend/
  fred_two_stream/
  models/
    early_fusion_best.pt
    two_fusion_best.pt
    late_fusion_rgb_best.pt
    late_fusion_event_best.pt
  team_b/
    results_summary/
    sample_detection_images/
  test_assets/
    sample_rgb.jpg
    sample_event.png
  Dockerfile
  docker-compose.yml
  requirements.txt
  requirements-docker.txt
  docs/
```

## Run with Docker

From this folder:

```powershell
docker compose up -d --build
```

Open:

```text
http://localhost:8000/docs
```

Stop:

```powershell
docker compose down
```

## Quick health check

Open:

```text
http://localhost:8000/api/health
```

Expected:

```json
{
  "status": "ok",
  "weights_available": {
    "early_fusion": true,
    "two_fusion": true,
    "late_fusion": true
  }
}
```

## Main API for frontend

```text
POST http://localhost:8000/api/detect
Content-Type: multipart/form-data
```

Required form fields:

```text
method: two_fusion, early_fusion, or late_fusion
rgb_file: RGB image
event_file: aligned Event image
```

Optional:

```text
confidence: 0.25
use_tracking: false
```

Recommended default:

```text
method = two_fusion
```

All methods require both `rgb_file` and `event_file`.

## Test files

Use these files for a first Swagger test:

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

## Team B comparison

```text
GET http://localhost:8000/api/compare
```

This returns offline/precomputed E2VID and FireNet summary data. These methods are not online `/api/detect` methods.

## More details

See:

```text
docs/API_HANDOFF.md
docs/TEAM_F_HANDOFF.md
docs/VALIDATION.md
```
