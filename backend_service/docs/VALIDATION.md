# Backend service validation record

Validation date: 6 July 2026

## Local environment

- Automated tests: 4/4 passed.
- Demo API smoke test: one paired sample, three methods.
- Requests: 3/3 successful.
- Device: NVIDIA GPU through local CUDA environment.
- Latest manual API test used `seq110_frame_000223`.

The stability run checks API execution, not detector accuracy. Formal precision,
recall, and mAP evaluation remains part of the model/evaluation work.

## Docker environment

- Docker Compose configuration parsed successfully.
- Folder tested: `backend_service/`
- Image: `backend_service-api:latest`.
- Image build: successful.
- Runtime mode: CPU, so no NVIDIA Container Toolkit is required.
- Container health status: not re-run in this standalone folder because the
  local Docker CLI later reported Windows named-pipe permission errors when
  creating a new container. The same backend code and model set were verified
  successfully through the rebuilt root Docker service immediately before this
  handoff refresh.

Verified locally with FastAPI TestClient from this standalone `backend_service`
folder:

- `GET /api/health`: HTTP 200; `early_fusion`, `two_fusion`, and
  `late_fusion` weights available.
- `GET /api/compare`: HTTP 200; E2VID and FireNet offline summaries available.
- `POST /api/detect method=two_fusion`: HTTP 200, one drone detection,
  confidence approximately 0.564.
- `POST /api/detect method=early_fusion`: HTTP 200, one drone detection,
  confidence approximately 0.524.
- `POST /api/detect method=late_fusion`: HTTP 200, one drone detection.
- Returned visualizations: HTTP 200 JPEG for all three methods.

The container remains available at `http://localhost:8000`. Stop it with:

```powershell
docker compose down
```
