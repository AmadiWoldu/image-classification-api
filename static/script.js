const API_BASE = "http://127.0.0.1:8000";

const CIFAR10_CLASSES = [
  "airplane", "automobile", "bird", "cat", "deer",
  "dog", "frog", "horse", "ship", "truck"
];

// ── DOM refs ──────────────────────────────────────────────
const statusBadge  = document.getElementById("statusBadge");
const statusText   = document.getElementById("statusText");
const dropZone     = document.getElementById("dropZone");
const fileInput    = document.getElementById("fileInput");
const uploadCard   = document.getElementById("uploadCard");
const previewCard  = document.getElementById("previewCard");
const previewImg   = document.getElementById("previewImg");
const resetBtn     = document.getElementById("resetBtn");
const predictBtn   = document.getElementById("predictBtn");
const btnText      = predictBtn.querySelector(".btn-text");
const btnLoader    = predictBtn.querySelector(".btn-loader");
const resultCard   = document.getElementById("resultCard");
const resultLabel  = document.getElementById("resultLabel");
const confVal      = document.getElementById("confVal");
const barFill      = document.getElementById("barFill");
const classesGrid  = document.getElementById("classesGrid");
const errorToast   = document.getElementById("errorToast");
const errorMsg     = document.getElementById("errorMsg");

let selectedFile = null;

// ── Health check ──────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      statusBadge.classList.add("online");
      statusBadge.classList.remove("offline");
      statusText.textContent = "Online";
    } else {
      throw new Error("Not OK");
    }
  } catch {
    statusBadge.classList.add("offline");
    statusBadge.classList.remove("online");
    statusText.textContent = "Offline";
  }
}
checkHealth();
setInterval(checkHealth, 30_000);

// ── Classes Grid init ─────────────────────────────────────
CIFAR10_CLASSES.forEach(cls => {
  const chip = document.createElement("div");
  chip.className = "class-chip";
  chip.dataset.cls = cls;
  chip.textContent = cls;
  classesGrid.appendChild(chip);
});

// ── Drag & Drop ───────────────────────────────────────────
dropZone.addEventListener("dragover", e => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
dropZone.addEventListener("click", (e) => {
  if (e.target !== fileInput && !e.target.classList.contains("btn-upload")) {
    fileInput.click();
  }
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

// ── File handler ──────────────────────────────────────────
function handleFile(file) {
  const allowed = ["image/jpeg", "image/png", "image/bmp", "image/gif", "image/webp"];
  if (!allowed.includes(file.type)) {
    showError("Unsupported file type. Please upload JPG, PNG, WEBP, BMP, or GIF.");
    return;
  }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    uploadCard.classList.add("hidden");
    previewCard.classList.remove("hidden");
    resultCard.classList.add("hidden");
    hideError();
  };
  reader.readAsDataURL(file);
}

// ── Reset ─────────────────────────────────────────────────
resetBtn.addEventListener("click", () => {
  selectedFile = null;
  fileInput.value = "";
  previewImg.src = "";
  previewCard.classList.add("hidden");
  resultCard.classList.add("hidden");
  uploadCard.classList.remove("hidden");
  hideError();
});

// ── Predict ───────────────────────────────────────────────
predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  setLoading(true);
  hideError();
  resultCard.classList.add("hidden");

  try {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Prediction failed." }));
      throw new Error(err.detail || "Prediction failed.");
    }

    const data = await res.json();
    showResult(data.prediction, data.confidence);
  } catch (err) {
    showError(err.message || "Could not reach the server.");
  } finally {
    setLoading(false);
  }
});

// ── Show Result ───────────────────────────────────────────
function showResult(prediction, confidence) {
  resultLabel.textContent = prediction;
  const pct = parseFloat(confidence.toFixed(2));
  confVal.textContent = `${pct}%`;

  resultCard.classList.remove("hidden");

  // Animate bar after paint
  requestAnimationFrame(() => {
    requestAnimationFrame(() => { barFill.style.width = `${pct}%`; });
  });

  // Highlight matching class chip
  document.querySelectorAll(".class-chip").forEach(chip => {
    chip.classList.toggle("active", chip.dataset.cls === prediction.toLowerCase());
  });
}

// ── Helpers ───────────────────────────────────────────────
function setLoading(loading) {
  predictBtn.disabled = loading;
  btnText.classList.toggle("hidden", loading);
  btnLoader.classList.toggle("hidden", !loading);
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorToast.classList.remove("hidden");
}

function hideError() {
  errorToast.classList.add("hidden");
}