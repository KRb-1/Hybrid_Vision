# Hybrid Vision Drone Detection Web Frontend

This file contains the web frontend for the **Hybrid Vision Drone Detection** demo. The website provides a user interface for running RGB-Event drone detection and for viewing comparison results of offline baseline methods.

The frontend is built with **Flask**, **HTML**, **CSS**, and **JavaScript**. It communicates with a separate FastAPI backend service through HTTP API calls.

---

## 1. Main Features

The website contains three main pages:

- **Home**: Introduction page for the Hybrid Vision project.
- **Detection**: Upload one RGB image and one synchronized Event frame image, select a fusion model, run detection, and display the returned bounding boxes and visualization image.
- **Comparison**: Display offline comparison statistics for E2VID and FireNet baselines using data loaded from the backend.

---

## 2. Project Structure

```text
HybridVision_Web/
├── app.py
├── templates/
│   ├── Home.html
│   ├── Detection.html
│   └── Comparison.html
├── static/
│   ├── Home.css
│   ├── Detection.css
│   ├── Detection.js
│   ├── Comparison.css
│   ├── Comparison.js
│   ├── nicepage.css
│   ├── nicepage.js
│   ├── jquery.js
│   └── images/
│       └── background.png
├── models/
│   └── detector.py
└── utils/
```

Important files:

| File | Description |
|---|---|
| `app.py` | Flask entry point. It serves the Home, Detection, and Comparison pages. |
| `templates/Home.html` | Landing page. |
| `templates/Detection.html` | Detection page UI. |
| `templates/Comparison.html` | Comparison page UI. |
| `static/Detection.js` | Calls `POST /api/detect` from the backend. |
| `static/Comparison.js` | Calls `GET /api/compare` from the backend. |
| `static/Detection.css` | Styles for the Detection page. |
| `static/Comparison.css` | Styles for the Comparison page. |
| `static/images/background.png` | Fixed background image used by Detection and Comparison pages. |

---

## 3. System Architecture

The website is only the frontend. The detection model and comparison data are provided by the backend service.

```text
Browser
  ↓
Flask Frontend
http://localhost:8888
  ↓ JavaScript fetch()
FastAPI Backend
http://localhost:8000
  ↓
Hybrid RGB-Event detection model / comparison data
```

Frontend pages:

```text
http://localhost:8888/             Home page
http://localhost:8888/detection    Detection page
http://localhost:8888/comparison   Comparison page
```

Backend API endpoints used by the frontend:

```text
GET  http://localhost:8000/api/health
POST http://localhost:8000/api/detect
GET  http://localhost:8000/api/compare
```

---

## 4. Prerequisites

Before running the website, make sure the following tools are installed:

### Required for frontend

- Python 3.9 or later
- Flask

### Required for backend

- Docker Desktop
- Docker Compose
- The backend service folder, usually named `backend_service`

The backend must be running before the Detection and Comparison pages can load API results.

---

## 5. Install Frontend Dependencies

Open a terminal or PowerShell inside the frontend folder:

```powershell
cd path\to\HybridVision_Web
```

Install Flask:

```powershell
pip install flask
```

If you use a virtual environment, create and activate it first:

```powershell
python -m venv venv
venv\Scripts\activate
pip install flask
```

---

## 6. Start the Backend Service

Open a new terminal or PowerShell and go to the backend folder:

```powershell
cd path\to\backend_service
```

Build and start the backend with Docker Compose:

```powershell
docker compose up -d --build
```

After the backend starts, check whether it is running:

```text
http://localhost:8000/api/health
```

A successful response should look similar to:

```json
{
  "status": "ok",
  "weights_available": {
    "early_fusion": true,
    "two_fusion": true
  }
}
```

You can also open the FastAPI documentation page:

```text
http://localhost:8000/docs
```

---

## 7. Start the Frontend Website

Open another terminal or PowerShell and go to the frontend folder:

```powershell
cd path\to\HybridVision_Web
```

Run the Flask app:

```powershell
python app.py
```

If `python` does not work on Windows, try:

```powershell
py app.py
```

The frontend should start at:

```text
http://localhost:8888
```

Open the website in your browser:

```text
http://localhost:8888/
```

---

## 8. How to Use the Detection Page

Open:

```text
http://localhost:8888/detection
```

The Detection page supports single-frame RGB-Event detection.

Steps:

1. Upload an **RGB Image**.
2. Upload a synchronized **Event Frame Image**.
3. Select a fusion model:
   - `two_fusion` recommended
   - `early_fusion`
4. Set the confidence threshold, for example `0.25`.
5. Click **Run Detection**.
6. The page will display:
   - RGB input preview
   - Event input preview
   - Detection output image returned by the backend
   - Detection class, confidence, `bbox_xyxy`, and `bbox_xywh`

The frontend sends a request to:

```text
POST http://localhost:8000/api/detect
```

The request contains:

```text
method: two_fusion or early_fusion
rgb_file: uploaded RGB image
event_file: uploaded Event frame image
confidence: confidence threshold
use_tracking: false
```

The backend returns a JSON response containing `detections` and `visualization_url`. The frontend displays the image by combining the backend address with the returned relative URL:

```javascript
resultImage.src = "http://localhost:8000" + result.visualization_url;
```

Important note: the current version supports one RGB image and one Event frame image. It does not directly support videos, image sequence ZIP files, or raw event streams.

---

## 9. Stop the System

### Stop the frontend

In the terminal running `python app.py`, press:

```text
Ctrl + C
```

### Stop the backend

Go to the backend folder:

```powershell
cd path\to\backend_service
```

Run:

```powershell
docker compose down
```

---

## 10. Quick Start Summary

Start backend:

```powershell
cd path\to\backend_service
docker compose up -d --build
```

Start frontend:

```powershell
cd path\to\HybridVision_Web
python app.py
```

Open website:

```text
http://localhost:8888
```

Use Detection page:

```text
http://localhost:8888/detection
```

Use Comparison page:

```text
http://localhost:8888/comparison
```
