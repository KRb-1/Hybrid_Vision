const BACKEND_URL = "http://localhost:8000";

const backendStatus = document.getElementById("backendStatus");
const refreshButton = document.getElementById("refreshButton");
const loadingSection = document.getElementById("loadingSection");
const errorSection = document.getElementById("errorSection");
const summarySection = document.getElementById("summarySection");
const sampleSection = document.getElementById("sampleSection");
const methodCards = document.getElementById("methodCards");
const sampleGallery = document.getElementById("sampleGallery");
const teamAInfo = document.getElementById("teamAInfo");

function setBackendStatus(text, type = "ok") {
  backendStatus.textContent = text;
  backendStatus.classList.toggle("ok", type === "ok");
  backendStatus.classList.toggle("error", type === "error");
}

function showLoading() {
  loadingSection.classList.remove("hidden");
  errorSection.classList.add("hidden");
  summarySection.classList.add("hidden");
  sampleSection.classList.add("hidden");
}

function showError(message) {
  loadingSection.classList.add("hidden");
  errorSection.classList.remove("hidden");
  errorSection.textContent = message;
}

function formatInteger(value) {
  return Number(value || 0).toLocaleString();
}

function formatCoverage(value) {
  const number = Number(value || 0);
  return `${(number * 100).toFixed(2)}%`;
}

function normalizeMethodName(methodName) {
  const lower = String(methodName || "").toLowerCase();
  if (lower === "e2vid") {
    return "E2VID";
  }
  if (lower === "firenet") {
    return "FireNet";
  }
  return methodName;
}

function getMethodEntries(data) {
  const methods = data && data.team_b && data.team_b.methods ? data.team_b.methods : {};
  return Object.entries(methods).sort(([nameA], [nameB]) => nameA.localeCompare(nameB));
}

function renderTeamAInfo(teamA) {
  if (!teamA) {
    teamAInfo.textContent = "Team A information is not available.";
    return;
  }

  const methods = Array.isArray(teamA.online_methods)
    ? teamA.online_methods.join(", ")
    : "not available";

  teamAInfo.innerHTML = `
    <strong>Online methods:</strong> ${methods}<br>
    <strong>Default method:</strong> ${teamA.default_method || "not available"}<br>
    <strong>Hybrid type:</strong> ${teamA.hybrid_type || "not available"}
  `;
}

function renderMethodCards(entries, executionMode) {
  methodCards.innerHTML = "";

  entries.forEach(([methodName, item]) => {
    const coverage = Number(item.detection_coverage || 0);
    const coveragePercent = Math.max(0, Math.min(100, coverage * 100));

    const card = document.createElement("article");
    card.className = "method-card";
    card.innerHTML = `
      <div class="method-card-header">
        <h3>${normalizeMethodName(methodName)}</h3>
        <span>${executionMode || "precomputed"}</span>
      </div>

      <div class="metric-grid">
        <div class="metric-box">
          <span class="metric-value">${formatInteger(item.sequences)}</span>
          <span class="metric-label">Sequences</span>
        </div>
        <div class="metric-box">
          <span class="metric-value">${formatInteger(item.reconstructed_frames)}</span>
          <span class="metric-label">Reconstructed frames</span>
        </div>
        <div class="metric-box">
          <span class="metric-value">${formatInteger(item.detected_frames)}</span>
          <span class="metric-label">Detected frames</span>
        </div>
        <div class="metric-box">
          <span class="metric-value">${formatCoverage(item.detection_coverage)}</span>
          <span class="metric-label">Coverage</span>
        </div>
      </div>

      <div class="coverage-bar" aria-label="Detection coverage">
        <div class="coverage-fill" style="width: ${coveragePercent}%;"></div>
      </div>

      <p class="metric-note">${item.metric_note || "Detection coverage is not mAP, precision, or recall."}</p>
    `;

    methodCards.appendChild(card);
  });
}

function renderSampleImages(entries) {
  sampleGallery.innerHTML = "";

  let totalImages = 0;

  entries.forEach(([methodName, item]) => {
    const images = Array.isArray(item.sample_images) ? item.sample_images.slice(0, 8) : [];
    if (images.length === 0) {
      return;
    }

    totalImages += images.length;

    const group = document.createElement("div");
    group.className = "sample-group";

    const title = document.createElement("h3");
    title.textContent = normalizeMethodName(methodName);
    group.appendChild(title);

    const imageGrid = document.createElement("div");
    imageGrid.className = "sample-image-grid";

    images.forEach((url, index) => {
      const fullUrl = `${BACKEND_URL}${url}`;
      const link = document.createElement("a");
      link.href = fullUrl;
      link.target = "_blank";
      link.rel = "noreferrer";

      const img = document.createElement("img");
      img.src = fullUrl;
      img.alt = `${normalizeMethodName(methodName)} sample ${index + 1}`;
      img.loading = "lazy";

      link.appendChild(img);
      imageGrid.appendChild(link);
    });

    group.appendChild(imageGrid);
    sampleGallery.appendChild(group);
  });

  if (totalImages === 0) {
    sampleGallery.innerHTML = `<p class="empty-message">No sample images were returned by the backend.</p>`;
  }
}

function renderComparison(data) {
  const entries = getMethodEntries(data);
  const teamB = data.team_b || {};
  const executionMode = teamB.execution_mode || "precomputed";

  renderTeamAInfo(data.team_a);

  loadingSection.classList.add("hidden");
  summarySection.classList.remove("hidden");
  sampleSection.classList.remove("hidden");

  if (entries.length === 0) {
    methodCards.innerHTML = `<p class="empty-message">No E2VID or FireNet summary data was returned.</p>`;
    comparisonTableBody.innerHTML = `
      <tr>
        <td colspan="6">No comparison data available.</td>
      </tr>
    `;
    sampleGallery.innerHTML = `<p class="empty-message">No sample images available.</p>`;
    return;
  }

  renderMethodCards(entries, executionMode);
  renderSampleImages(entries);
}

async function loadComparison() {
  showLoading();
  refreshButton.disabled = true;
  refreshButton.textContent = "Loading...";

  try {
    const response = await fetch(`${BACKEND_URL}/api/compare`);
    const data = await response.json().catch(() => null);

    if (!response.ok) {
      const detail = data && data.detail ? data.detail : `HTTP ${response.status}`;
      throw new Error(detail);
    }

    setBackendStatus(`Backend connected: ${data.status || "success"}`, "ok");
    renderComparison(data);
  } catch (error) {
    console.error(error);
    setBackendStatus("Backend not connected. Please start FastAPI service on port 8000.", "error");
    showError(`Failed to load comparison data: ${error.message}`);
  } finally {
    refreshButton.disabled = false;
    refreshButton.textContent = "Refresh Comparison";
  }
}

refreshButton.addEventListener("click", loadComparison);
loadComparison();
