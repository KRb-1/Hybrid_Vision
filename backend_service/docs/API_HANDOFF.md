# Frontend hand-off contract

The Swagger page at `/docs` is a development tool, not the final graphical user
interface. Team F should call the API below from the project web interface.

## Required endpoints

### `GET /api/health`

Use on application startup. A successful response has `status: "ok"` and reports
whether the `early_fusion`, `two_fusion`, and `late_fusion` weights are present.

### `POST /api/detect`

Content type: `multipart/form-data`.

Fields:

- `method`: `two_fusion`, `early_fusion`, or `late_fusion`. Default is `two_fusion`.
- `rgb_file`: required.
- `event_file`: required.
- `confidence`: optional float from 0 to 1; default `0.25`.
- `use_tracking`: currently must be `false`.

This version accepts one aligned image pair. Supported image extensions are
JPG, PNG, BMP, WebP, TIFF, and TIF. The Event input is an already generated
event frame, not a raw event stream. `late_fusion` is an online baseline that
runs separate RGB/Event detectors and merges boxes with NMS.

Each detection contains pixel `bbox_xyxy`, pixel `bbox_xywh`, class ID/name,
confidence, and source. Prefix `visualization_url` with the backend origin, for
example `http://localhost:8000/results/.../detection.jpg`.

### `GET /api/compare`

Optional report-level endpoint for Team B's precomputed E2VID/FireNet summary.
Those reconstruction methods are not online detection methods.

## Frontend responsibilities

- Select matching RGB and Event files.
- Submit multipart requests and display progress/errors.
- Show both inputs and the returned visualization.
- Provide image/video playback in the final UI.

Sequence/video upload is not part of the current single-frame API. Team E and F
must agree on ZIP, multi-file, or video input before adding playback processing.
