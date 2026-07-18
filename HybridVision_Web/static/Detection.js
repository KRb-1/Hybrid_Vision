const BACKEND_URL = "http://localhost:8000";

const detectForm = document.getElementById("detectForm");
const inputTypeSelect = document.getElementById("inputTypeSelect");

const imageInputs = document.getElementById("imageInputs");
const sequenceInputs = document.getElementById("sequenceInputs");

const rgbFileInput = document.getElementById("rgbFile");
const eventFileInput = document.getElementById("eventFile");
const rgbZipInput = document.getElementById("rgbZip");
const eventZipInput = document.getElementById("eventZip");

const methodSelect = document.getElementById("methodSelect");
const confidenceInput = document.getElementById("confidenceInput");
const fpsInput = document.getElementById("fpsInput");
const maxFramesInput = document.getElementById("maxFramesInput");
const fpsGroup = document.getElementById("fpsGroup");
const maxFramesGroup = document.getElementById("maxFramesGroup");

const detectButton = document.getElementById("detectButton");
const backendStatus = document.getElementById("backendStatus");
const message = document.getElementById("message");

const previewSection = document.getElementById("previewSection");
const resultSection = document.getElementById("resultSection");
const videoResultSection = document.getElementById("videoResultSection");

const rgbPreview = document.getElementById("rgbPreview");
const eventPreview = document.getElementById("eventPreview");
const resultImage = document.getElementById("resultImage");
const summaryBox = document.getElementById("summaryBox");
const detectionsTableBody = document.querySelector("#detectionsTable tbody");

const rgbInputVideo = document.getElementById("rgbInputVideo");
const eventInputVideo = document.getElementById("eventInputVideo");
const outputDetectionVideo = document.getElementById("outputDetectionVideo");

const playAllButton = document.getElementById("playAllButton");
const pauseAllButton = document.getElementById("pauseAllButton");
const restartAllButton = document.getElementById("restartAllButton");
const playbackRateSelect = document.getElementById("playbackRateSelect");

const sequenceSummaryBox = document.getElementById("sequenceSummaryBox");

const videoElements = [rgbInputVideo, eventInputVideo, outputDetectionVideo];
let isSyncingVideos = false;

function setMessage(text, type = "error") {
  message.textContent = text;
  message.classList.toggle("success", type === "success");
}

function clearMessage() {
  message.textContent = "";
  message.classList.remove("success");
}

function formatNumber(value) {
  if (typeof value !== "number") {
    return "-";
  }
  return value.toFixed(3);
}

function formatBBox(values) {
  if (!Array.isArray(values)) {
    return "-";
  }
  return `[${values.map((v) => Number(v).toFixed(1)).join(", ")}]`;
}

function hideAllResults() {
  previewSection.classList.add("hidden");
  resultSection.classList.add("hidden");
  videoResultSection.classList.add("hidden");
}

function updateInputMode() {
  const inputType = inputTypeSelect.value;
  clearMessage();
  hideAllResults();

  if (inputType === "image") {
    imageInputs.classList.remove("hidden");
    sequenceInputs.classList.add("hidden");
    fpsGroup.classList.add("hidden");
    maxFramesGroup.classList.add("hidden");

    rgbFileInput.required = true;
    eventFileInput.required = true;
    rgbZipInput.required = false;
    eventZipInput.required = false;

    detectButton.textContent = "Run Detection";
  } else {
    imageInputs.classList.add("hidden");
    sequenceInputs.classList.remove("hidden");
    fpsGroup.classList.remove("hidden");
    maxFramesGroup.classList.remove("hidden");

    rgbFileInput.required = false;
    eventFileInput.required = false;
    rgbZipInput.required = true;
    eventZipInput.required = true;

    detectButton.textContent = "Run Sequence Detection";
  }
}

function previewInputImage(input, imageElement) {
  const file = input.files[0];

  if (!file) {
    imageElement.removeAttribute("src");
    return;
  }

  imageElement.src = URL.createObjectURL(file);
  previewSection.classList.remove("hidden");
}

async function checkBackendHealth() {
  try {
    const response = await fetch(`${BACKEND_URL}/api/health`);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    backendStatus.textContent = `Backend connected: ${data.status}`;
    backendStatus.classList.add("ok");
    backendStatus.classList.remove("error");
  } catch (error) {
    backendStatus.textContent = "Backend not connected. Please start FastAPI service on port 8000.";
    backendStatus.classList.add("error");
    backendStatus.classList.remove("ok");
  }
}

function renderSingleImageResults(result) {
  const detections = result.detections || [];

  resultSection.classList.remove("hidden");
  videoResultSection.classList.add("hidden");
  previewSection.classList.remove("hidden");

  if (rgbFileInput.files[0]) {
    rgbPreview.src = URL.createObjectURL(rgbFileInput.files[0]);
  }

  if (eventFileInput.files[0]) {
    eventPreview.src = URL.createObjectURL(eventFileInput.files[0]);
  }

  if (result.visualization_url) {
    resultImage.src = `${BACKEND_URL}${result.visualization_url}?t=${Date.now()}`;
  } else {
    resultImage.removeAttribute("src");
  }

  summaryBox.innerHTML = `
    <strong>Status:</strong> ${result.status || "success"}<br>
    <strong>Method:</strong> ${result.method || methodSelect.value}<br>
    <strong>Execution Mode:</strong> ${result.execution_mode || "online"}<br>
    <strong>Number of detections:</strong> ${detections.length}
  `;

  detectionsTableBody.innerHTML = "";

  if (detections.length === 0) {
    detectionsTableBody.innerHTML = `
      <tr>
        <td colspan="6">No drone detected.</td>
      </tr>
    `;
    return;
  }

  detections.forEach((det, index) => {
    const row = document.createElement("tr");

    row.innerHTML = `
      <td>${index + 1}</td>
      <td>${det.class_name || det.class_id || "drone"}</td>
      <td>${formatNumber(det.confidence)}</td>
      <td>${det.source || result.method || "-"}</td>
      <td>${formatBBox(det.bbox_xyxy)}</td>
      <td>${formatBBox(det.bbox_xywh)}</td>
    `;

    detectionsTableBody.appendChild(row);
  });
}

function syncVideosTo(sourceVideo) {
  if (isSyncingVideos) {
    return;
  }

  isSyncingVideos = true;

  videoElements.forEach((video) => {
    if (video !== sourceVideo && Number.isFinite(sourceVideo.currentTime)) {
      if (Math.abs(video.currentTime - sourceVideo.currentTime) > 0.15) {
        video.currentTime = sourceVideo.currentTime;
      }
    }

    video.playbackRate = sourceVideo.playbackRate;
  });

  window.setTimeout(() => {
    isSyncingVideos = false;
  }, 80);
}

function playAllVideos(sourceVideo = outputDetectionVideo) {
  syncVideosTo(sourceVideo);

  videoElements.forEach((video) => {
    video.play().catch(() => {
      // Browser may block autoplay. User can click Play All manually.
    });
  });
}

function pauseAllVideos() {
  if (isSyncingVideos) {
    return;
  }

  isSyncingVideos = true;

  videoElements.forEach((video) => {
    video.pause();
  });

  window.setTimeout(() => {
    isSyncingVideos = false;
  }, 80);
}

function restartAllVideos() {
  isSyncingVideos = true;

  videoElements.forEach((video) => {
    video.pause();
    video.currentTime = 0;
  });

  window.setTimeout(() => {
    isSyncingVideos = false;
    playAllVideos(outputDetectionVideo);
  }, 120);
}

function setupVideoSynchronization() {
  videoElements.forEach((video) => {
    video.loop = true;
    video.muted = true;

    video.addEventListener("play", () => {
      if (!isSyncingVideos) {
        playAllVideos(video);
      }
    });

    video.addEventListener("pause", () => {
      if (!isSyncingVideos && !video.ended) {
        pauseAllVideos();
      }
    });

    video.addEventListener("seeking", () => {
      syncVideosTo(video);
    });

    video.addEventListener("ratechange", () => {
      if (!isSyncingVideos) {
        videoElements.forEach((item) => {
          item.playbackRate = video.playbackRate;
        });
      }
    });
  });

  outputDetectionVideo.addEventListener("timeupdate", () => {
    if (!isSyncingVideos) {
      syncVideosTo(outputDetectionVideo);
    }
  });
}

function renderSequenceResults(result) {
  resultSection.classList.add("hidden");
  previewSection.classList.add("hidden");
  videoResultSection.classList.remove("hidden");

  const cacheBuster = `?t=${Date.now()}`;

  rgbInputVideo.src = `${BACKEND_URL}${result.rgb_video_url}${cacheBuster}`;
  eventInputVideo.src = `${BACKEND_URL}${result.event_video_url}${cacheBuster}`;
  outputDetectionVideo.src = `${BACKEND_URL}${result.output_video_url}${cacheBuster}`;

  videoElements.forEach((video) => {
    video.load();
    video.currentTime = 0;
    video.playbackRate = Number(playbackRateSelect.value);
  });

  sequenceSummaryBox.innerHTML = `
    <strong>Status:</strong> ${result.status || "success"}<br>
    <strong>Method:</strong> ${result.method || methodSelect.value}<br>
    <strong>Frame Count:</strong> ${result.frame_count || 0}<br>
    <strong>FPS:</strong> ${result.fps || fpsInput.value}<br>
    <strong>Total Detections:</strong> ${result.total_detections || 0}
  `;

  setTimeout(() => {
    playAllVideos(outputDetectionVideo);
  }, 600);
}

async function runSingleImageDetection() {
  const rgbFile = rgbFileInput.files[0];
  const eventFile = eventFileInput.files[0];

  if (!rgbFile || !eventFile) {
    setMessage("Please upload both an RGB image and an Event image.");
    return;
  }

  const formData = new FormData();
  formData.append("method", methodSelect.value);
  formData.append("rgb_file", rgbFile);
  formData.append("event_file", eventFile);
  formData.append("confidence", confidenceInput.value || "0.25");
  formData.append("use_tracking", "false");

  const response = await fetch(`${BACKEND_URL}/api/detect`, {
    method: "POST",
    body: formData,
  });

  const result = await response.json().catch(() => null);

  if (!response.ok) {
    const detail = result && result.detail ? result.detail : `HTTP ${response.status}`;
    throw new Error(detail);
  }

  renderSingleImageResults(result);
}

async function runSequenceDetection() {
  const rgbZip = rgbZipInput.files[0];
  const eventZip = eventZipInput.files[0];

  if (!rgbZip || !eventZip) {
    setMessage("Please upload both RGB Frames ZIP and Event Frames ZIP.");
    return;
  }

  const formData = new FormData();
  formData.append("method", methodSelect.value);
  formData.append("rgb_zip", rgbZip);
  formData.append("event_zip", eventZip);
  formData.append("confidence", confidenceInput.value || "0.25");
  formData.append("fps", fpsInput.value || "30");
  formData.append("max_frames", maxFramesInput.value || "300");

  const response = await fetch(`${BACKEND_URL}/api/detect-sequence`, {
    method: "POST",
    body: formData,
  });

  const result = await response.json().catch(() => null);

  if (!response.ok) {
    const detail = result && result.detail ? result.detail : `HTTP ${response.status}`;
    throw new Error(detail);
  }

  renderSequenceResults(result);
}

async function runDetection(event) {
  event.preventDefault();
  clearMessage();
  hideAllResults();

  const inputType = inputTypeSelect.value;

  detectButton.disabled = true;
  detectButton.textContent = inputType === "sequence" ? "Processing Sequence..." : "Running...";
  setMessage("Running detection. Please wait...", "success");

  try {
    if (inputType === "image") {
      await runSingleImageDetection();
      setMessage("Detection completed successfully.", "success");
    } else {
      await runSequenceDetection();
      setMessage("Sequence detection completed. Videos are synchronized for playback.", "success");
    }
  } catch (error) {
    console.error(error);
    setMessage(`Detection failed: ${error.message}`);
  } finally {
    detectButton.disabled = false;
    detectButton.textContent = inputType === "sequence" ? "Run Sequence Detection" : "Run Detection";
  }
}

inputTypeSelect.addEventListener("change", updateInputMode);

rgbFileInput.addEventListener("change", () => {
  previewInputImage(rgbFileInput, rgbPreview);
});

eventFileInput.addEventListener("change", () => {
  previewInputImage(eventFileInput, eventPreview);
});

detectForm.addEventListener("submit", runDetection);

playAllButton.addEventListener("click", () => playAllVideos(outputDetectionVideo));
pauseAllButton.addEventListener("click", pauseAllVideos);
restartAllButton.addEventListener("click", restartAllVideos);

playbackRateSelect.addEventListener("change", () => {
  const rate = Number(playbackRateSelect.value);

  videoElements.forEach((video) => {
    video.playbackRate = rate;
  });
});

setupVideoSynchronization();
updateInputMode();
checkBackendHealth();