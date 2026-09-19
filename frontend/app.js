/* VisionID dashboard — vanilla JS SPA, no build step required. */

const API = "";

/* ---------------------------------------------------------------- utils */

function el(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
}

function fmtTime(iso) {
  try {
    const d = new Date(iso);
    return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  } catch {
    return iso || "";
  }
}

function toast(msg) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toast._h);
  toast._h = setTimeout(() => t.classList.remove("show"), 2600);
}

async function api(path, options = {}) {
  const res = await fetch(API + path, options);
  let data = null;
  try { data = await res.json(); } catch { /* no body */ }
  if (!res.ok) {
    const detail = data && data.detail;
    let msg = `Request failed (${res.status})`;
    if (typeof detail === "string") msg = detail;
    else if (detail && typeof detail === "object") {
      msg = detail.message || JSON.stringify(detail);
      if (detail.similarity != null) msg += ` Similarity: ${Math.round(Number(detail.similarity) * 100)}%.`;
    }
    throw new Error(msg);
  }
  return data;
}

/* ---------------------------------------------------------------- theme */

(function initTheme() {
  const saved = "dark";
  document.documentElement.setAttribute("data-theme", saved);
  document.body.setAttribute("data-theme", saved);
  document.getElementById("theme-toggle").textContent = "☀";
})();

document.getElementById("theme-toggle").addEventListener("click", () => {
  const current = document.body.getAttribute("data-theme") === "light" ? "dark" : "light";
  document.body.setAttribute("data-theme", current);
  document.getElementById("theme-toggle").textContent = current === "light" ? "🌙" : "☀";
});

/* ---------------------------------------------------------------- system status poll */

async function pollSystem() {
  try {
    const h = await api("/api/health");
    document.getElementById("sys-backend").textContent = "ONLINE";
    document.getElementById("sys-backend").className = "ok";
    document.getElementById("sys-db").textContent = "CONNECTED";
    document.getElementById("sys-db").className = "ok";
    document.getElementById("sys-model").textContent = "LOADED";
    document.getElementById("sys-model").className = "ok";
  } catch {
    document.getElementById("sys-backend").textContent = "OFFLINE";
    document.getElementById("sys-backend").className = "bad";
  }
}
pollSystem();
setInterval(pollSystem, 15000);

/* ---------------------------------------------------------------- router */

const routes = {
  dashboard: renderDashboard,
  recognize: renderRecognize,
  enroll: renderEnroll,
  people: renderPeople,
  evaluation: renderEvaluation,
  notes: renderNotes,
};

const crumbs = {
  dashboard: "/ command center",
  recognize: "/ inference console · live scan",
  enroll: "/ identity enrollment",
  people: "/ people directory",
  evaluation: "/ evaluation lab · validation",
  notes: "/ model notes · explainability",
};

function currentRoute() {
  const hash = location.hash.replace("#/", "") || "dashboard";
  return routes[hash] ? hash : "dashboard";
}

async function router() {
  const route = currentRoute();
  document.querySelectorAll(".nav a").forEach((a) => {
    a.classList.toggle("active", a.dataset.route === route);
  });
  document.getElementById("crumb").textContent = crumbs[route];
  const content = document.getElementById("content");
  content.innerHTML = `<div class="empty">Loading…</div>`;
  try {
    await routes[route](content);
  } catch (err) {
    content.innerHTML = `<div class="empty">Something went wrong: ${err.message}</div>`;
  }
}

window.addEventListener("hashchange", router);
window.addEventListener("DOMContentLoaded", router);

/* ---------------------------------------------------------------- DASHBOARD */

async function renderDashboard(content) {
  content.innerHTML = `
    <div class="hero">
      <div class="eyebrow">VisionID / Control Plane</div>
      <h1>See the signal.<br/><span class="grad">Know the face.</span></h1>
      <p>A recognition workspace for authorized identity checks, explainable similarity decisions, and accountable evaluation. The same UI can run with persistent local SQLite or temporary hosted demo storage.</p>
      <div class="hero-actions">
        <a class="btn btn-primary" href="#/recognize">◎ Start recognition</a>
        <a class="btn" href="#/enroll">✚ Enroll identity</a>
      </div>
    </div>

    <div class="grid grid-4" id="stat-cards">
      <div class="empty">Loading stats…</div>
    </div>

    <div class="two-col" style="margin-top:16px">
      <div class="card">
        <div class="section-title">Recent recognition activity</div>
        <div class="section-sub" id="activity-count">Latest attempts</div>
        <div id="activity-list"></div>
      </div>

      <div class="card threshold-block">
        <div class="section-title">Threshold guardrail</div>
        <div class="section-sub">Decision policy</div>
        <div class="thr-value" id="thr-value">—</div>
        <div class="thr-track"><div class="thr-fill" id="thr-fill" style="width:0%"></div></div>
        <p class="thr-caption">A face is accepted only when its cosine similarity to the
        nearest enrolled identity clears this bar. Tune it from the Recognize page or
        via <code>/api/settings/threshold</code>.</p>
        <a class="btn btn-block" href="#/evaluation">Open evaluation lab →</a>
      </div>
    </div>
  `;

  const [stats, activity] = await Promise.all([
    api("/api/stats"),
    api("/api/activity?limit=8"),
  ]);

  document.getElementById("stat-cards").innerHTML = `
    ${statCard("👥", stats.enrolled_people, "Enrolled people", "active identities")}
    ${statCard("◎", stats.recognition_attempts, "Recognition attempts", "all processed scans")}
    ${statCard("✔", stats.successful_matches, "Successful matches", stats.acceptance_rate + "% acceptance rate")}
    ${statCard("⚠", stats.unknown_faces, "Unknown faces", "rejected by threshold")}
  `;

  document.getElementById("thr-value").textContent = Math.round(stats.threshold * 100) + "%";
  document.getElementById("thr-fill").style.width = Math.round(stats.threshold * 100) + "%";

  const list = document.getElementById("activity-list");
  if (!activity.activity.length) {
    list.innerHTML = `<div class="empty">No recognition attempts yet. Try the Recognize page.</div>`;
  } else {
    list.innerHTML = activity.activity.map((a) => `
      <div class="activity-item">
        <div class="activity-left">
          <span class="dot ${a.known ? "dot-known" : "dot-unknown"}"></span>
          <div>
            <div class="activity-name">${a.known ? escapeHtml(a.name) : "Unknown face"}</div>
            <div class="activity-meta">${fmtTime(a.timestamp)} · ${a.source} · ${a.face_count} face${a.face_count === 1 ? "" : "s"}</div>
          </div>
        </div>
        <div class="activity-score" style="color:${a.known ? "var(--good)" : "var(--warn)"}">${Math.round(a.similarity * 100)}%</div>
      </div>
    `).join("");
  }
}

function statCard(icon, value, label, sub) {
  return `
    <div class="card stat-card">
      <div class="stat-top"><div class="stat-icon">${icon}</div></div>
      <div class="stat-value">${value}</div>
      <div class="stat-label">${label}</div>
      <div class="stat-sub">${sub}</div>
    </div>
  `;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ---------------------------------------------------------------- RECOGNIZE */

let recognizeState = { stream: null, timer: null, mode: "upload" };

async function renderRecognize(content) {
  stopWebcamLoop();
  const threshold = (await api("/api/settings/threshold")).threshold;

  content.innerHTML = `
    <div class="eyebrow">Inference console / Live scan</div>
    <h1 class="page-title">Recognize a face</h1>
    <p class="page-desc">Upload an image or start your webcam for continuous live matching.
    Every detected face is matched independently and can be rejected as <b>Unknown</b>.</p>

    <div class="tabs" id="rec-tabs">
      <div class="tab active" data-mode="upload">Upload image</div>
      <div class="tab" data-mode="webcam">Live webcam</div>
    </div>

    <div class="two-col">
      <div class="card">
        <div class="section-title" id="input-title">Input signal</div>
        <div class="section-sub">JPG / PNG / WEBP</div>
        <div id="rec-input-area"></div>
      </div>

      <div class="card">
        <div class="section-title">Decision output</div>
        <div class="section-sub">Per-face similarity + threshold decision</div>
        <div id="rec-output" class="empty">Awaiting an input signal.</div>
      </div>
    </div>

    <div class="card" style="margin-top:16px">
      <div class="section-title">Matching threshold</div>
      <div class="section-sub">Live-adjustable — applies to every recognition immediately</div>
      <div style="display:flex;align-items:center;gap:14px">
        <input type="range" min="0" max="100" value="${Math.round(threshold * 100)}" id="rec-threshold" style="flex:1" />
        <b id="rec-threshold-val" style="min-width:48px;text-align:right">${Math.round(threshold * 100)}%</b>
      </div>
    </div>
  `;

  document.getElementById("rec-threshold").addEventListener("input", async (e) => {
    document.getElementById("rec-threshold-val").textContent = e.target.value + "%";
  });
  document.getElementById("rec-threshold").addEventListener("change", async (e) => {
    const value = Number(e.target.value) / 100;
    await api("/api/settings/threshold", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ value }),
    });
    toast(`Threshold set to ${e.target.value}%`);
  });

  document.querySelectorAll("#rec-tabs .tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll("#rec-tabs .tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      recognizeState.mode = tab.dataset.mode;
      stopWebcamLoop();
      if (tab.dataset.mode === "upload") renderUploadInput();
      else renderWebcamInput();
    });
  });

  renderUploadInput();
}

function renderUploadInput() {
  const area = document.getElementById("rec-input-area");
  area.innerHTML = `
    <div class="dropzone" id="dz">
      <div class="dz-icon">⇪</div>
      <div class="dz-title">Drop an image or browse</div>
      <div class="dz-sub">Multi-face · JPG / PNG / WEBP</div>
      <input type="file" id="dz-input" accept="image/*" hidden />
    </div>
  `;
  const dz = document.getElementById("dz");
  const input = document.getElementById("dz-input");
  dz.addEventListener("click", () => input.click());
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("drag");
    if (e.dataTransfer.files[0]) handleRecognizeFile(e.dataTransfer.files[0]);
  });
  input.addEventListener("change", () => {
    if (input.files[0]) handleRecognizeFile(input.files[0]);
  });
}

async function handleRecognizeFile(file) {
  const area = document.getElementById("rec-input-area");
  const url = URL.createObjectURL(file);
  area.innerHTML = `
    <div class="preview-wrap" id="preview-wrap">
      <img id="preview-img" src="${url}" />
      <canvas id="preview-canvas"></canvas>
    </div>
    <button class="btn btn-block" style="margin-top:12px" id="rescan-btn">↺ Choose another image</button>
  `;
  document.getElementById("rescan-btn").addEventListener("click", renderUploadInput);
  const img = document.getElementById("preview-img");
  img.onload = async () => {
    setOutputLoading();
    try {
      const result = await recognizeImage(file, "upload");
      drawBoxes(document.getElementById("preview-canvas"), img, result);
      renderRecognitionOutput(result);
    } catch (err) {
      document.getElementById("rec-output").innerHTML = `<div class="empty">${err.message}</div>`;
    }
  };
}

function setOutputLoading() {
  document.getElementById("rec-output").innerHTML = `<div class="empty"><span class="spinner"></span>&nbsp; Running detection + matching…</div>`;
}

async function recognizeImage(fileOrBlob, source) {
  const form = new FormData();
  form.append("image", fileOrBlob, "frame.jpg");
  return api(`/api/recognize?source=${source}`, { method: "POST", body: form });
}

function drawBoxes(canvas, imgEl, result) {
  const rect = imgEl.getBoundingClientRect();
  canvas.width = imgEl.clientWidth;
  canvas.height = imgEl.clientHeight;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!result.faces || !result.image_width) return;
  const scaleX = canvas.width / result.image_width;
  const scaleY = canvas.height / result.image_height;
  result.faces.forEach((f) => {
    const [x1, y1, x2, y2] = f.box;
    const color = f.known ? "#4ade80" : "#fbbf24";
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);
    const label = f.known ? `${f.name} ${Math.round(f.similarity * 100)}%` : "Unknown";
    ctx.font = "600 12px Inter, sans-serif";
    const textW = ctx.measureText(label).width + 12;
    ctx.fillStyle = color;
    ctx.fillRect(x1 * scaleX, Math.max(0, y1 * scaleY - 22), textW, 22);
    ctx.fillStyle = "#0a0a12";
    ctx.fillText(label, x1 * scaleX + 6, y1 * scaleY - 6);
  });
}

function renderRecognitionOutput(result) {
  const out = document.getElementById("rec-output");
  if (!result.faces || !result.faces.length) {
    out.innerHTML = `<div class="empty">No face detected in this frame.</div>`;
    return;
  }
  out.innerHTML = result.faces.map((f, i) => `
    <div class="activity-item">
      <div class="activity-left">
        <span class="dot ${f.known ? "dot-known" : "dot-unknown"}"></span>
        <div>
          <div class="activity-name">${f.known ? escapeHtml(f.name) : `Face ${i + 1} · Unknown`}</div>
          <div class="activity-meta">detection ${Math.round(f.detection_confidence * 100)}% · similarity ${Math.round(f.similarity * 100)}%</div>
        </div>
      </div>
      <span class="badge ${f.known ? "badge-known" : "badge-unknown"}">${f.known ? "MATCH" : "UNKNOWN"}</span>
    </div>
  `).join("");
}

function renderWebcamInput() {
  const area = document.getElementById("rec-input-area");
  area.innerHTML = `
    <div class="preview-wrap" id="preview-wrap">
      <video id="webcam-video" autoplay muted playsinline></video>
      <canvas id="preview-canvas"></canvas>
    </div>
    <div style="display:flex;gap:10px;margin-top:12px">
      <button class="btn btn-primary btn-block" id="webcam-toggle">▶ Start camera</button>
    </div>
  `;
  document.getElementById("webcam-toggle").addEventListener("click", toggleWebcam);
}

async function toggleWebcam() {
  const btn = document.getElementById("webcam-toggle");
  if (recognizeState.stream) {
    stopWebcamLoop();
    btn.textContent = "▶ Start camera";
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 960, height: 540 } });
    recognizeState.stream = stream;
    const video = document.getElementById("webcam-video");
    video.srcObject = stream;
    btn.textContent = "■ Stop camera";
    startWebcamLoop();
  } catch (err) {
    toast("Camera access denied or unavailable.");
  }
}

function startWebcamLoop() {
  const video = document.getElementById("webcam-video");
  const canvas = document.getElementById("preview-canvas");
  const capture = document.createElement("canvas");
  let busy = false;

  recognizeState.timer = setInterval(async () => {
    if (busy || !video.videoWidth) return;
    busy = true;
    capture.width = video.videoWidth;
    capture.height = video.videoHeight;
    capture.getContext("2d").drawImage(video, 0, 0);
    capture.toBlob(async (blob) => {
      try {
        const result = await recognizeImage(blob, "webcam");
        drawBoxes(canvas, video, result);
        renderRecognitionOutput(result);
      } catch { /* keep looping even if a frame fails */ }
      busy = false;
    }, "image/jpeg", 0.85);
  }, 900);
}

function stopWebcamLoop() {
  if (recognizeState.timer) clearInterval(recognizeState.timer);
  recognizeState.timer = null;
  if (recognizeState.stream) {
    recognizeState.stream.getTracks().forEach((t) => t.stop());
  }
  recognizeState.stream = null;
}

/* ---------------------------------------------------------------- ENROLL */

let stagedFiles = [];

async function renderEnroll(content) {
  stagedFiles = [];
  content.innerHTML = `
    <div class="eyebrow">Identity enrollment</div>
    <h1 class="page-title">Enroll an identity</h1>
    <p class="page-desc">Create a biometric reference from one or more authorized images.
    Each image must contain exactly one detectable face; embeddings are averaged for robustness.</p>

    <div class="two-col">
      <div class="card">
        <div class="section-title">Identity details</div>
        <div class="field"><label>Person name</label><input type="text" id="f-name" placeholder="e.g. Alex Morgan" /></div>
        <div class="field"><label>Person ID (optional)</label><input type="text" id="f-id" placeholder="e.g. P-001" /></div>
        <div class="field"><label>Description (optional)</label><textarea id="f-desc" placeholder="Context for authorized operators"></textarea></div>
        <div class="note-box">All persistent application data — including normalized embeddings and enrollment images — is stored in the SQLite database <code>backend/data/face_recognition.db</code>. No browser localStorage or JSON database is used.</div>
      </div>

      <div class="card">
        <div class="section-title">Reference images</div>
        <div class="section-sub">Add up to 8 angles for better robustness — <span id="img-count">0</span>/8</div>
        <div class="dropzone" id="enroll-dz">
          <div class="dz-icon">⇪</div>
          <div class="dz-title">Drop image files or browse</div>
          <div class="dz-sub">JPG · PNG · WEBP</div>
          <input type="file" id="enroll-input" accept="image/*" multiple hidden />
        </div>
        <div class="thumb-row" id="thumb-row"></div>
        <div style="display:flex;gap:10px;margin-top:16px">
          <button class="btn" id="enroll-webcam-btn">📷 Use camera</button>
          <button class="btn btn-primary" style="flex:1" id="enroll-submit">✚ Enroll person</button>
        </div>
        <div id="enroll-webcam-area" style="margin-top:12px"></div>
      </div>
    </div>
  `;

  const dz = document.getElementById("enroll-dz");
  const input = document.getElementById("enroll-input");
  dz.addEventListener("click", () => input.click());
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault(); dz.classList.remove("drag");
    addStagedFiles([...e.dataTransfer.files]);
  });
  input.addEventListener("change", () => addStagedFiles([...input.files]));

  document.getElementById("enroll-webcam-btn").addEventListener("click", openEnrollWebcam);
  document.getElementById("enroll-submit").addEventListener("click", submitEnrollment);
}

function addStagedFiles(files) {
  const room = 8 - stagedFiles.length;
  files.slice(0, room).forEach((f) => stagedFiles.push(f));
  renderThumbs();
}

function renderThumbs() {
  document.getElementById("img-count").textContent = stagedFiles.length;
  const row = document.getElementById("thumb-row");
  row.innerHTML = stagedFiles.map((f, i) => `
    <div class="thumb">
      <img src="${URL.createObjectURL(f)}" />
      <button class="rm" data-i="${i}">✕</button>
    </div>
  `).join("");
  row.querySelectorAll(".rm").forEach((btn) => {
    btn.addEventListener("click", () => {
      stagedFiles.splice(Number(btn.dataset.i), 1);
      renderThumbs();
    });
  });
}

async function openEnrollWebcam() {
  const area = document.getElementById("enroll-webcam-area");
  if (area.dataset.open === "1") {
    area.innerHTML = "";
    area.dataset.open = "0";
    if (area._stream) area._stream.getTracks().forEach((t) => t.stop());
    return;
  }
  area.dataset.open = "1";
  area.innerHTML = `
    <div class="preview-wrap"><video id="enroll-video" autoplay muted playsinline></video></div>
    <button class="btn btn-block" style="margin-top:10px" id="enroll-capture">◉ Capture frame</button>
  `;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    area._stream = stream;
    document.getElementById("enroll-video").srcObject = stream;
    document.getElementById("enroll-capture").addEventListener("click", () => {
      const video = document.getElementById("enroll-video");
      const c = document.createElement("canvas");
      c.width = video.videoWidth; c.height = video.videoHeight;
      c.getContext("2d").drawImage(video, 0, 0);
      c.toBlob((blob) => {
        const file = new File([blob], `capture-${Date.now()}.jpg`, { type: "image/jpeg" });
        addStagedFiles([file]);
        toast("Frame captured");
      }, "image/jpeg", 0.9);
    });
  } catch {
    toast("Camera access denied or unavailable.");
  }
}

async function submitEnrollment() {
  const name = document.getElementById("f-name").value.trim();
  if (!name) { toast("Please enter a person name."); return; }
  if (!stagedFiles.length) { toast("Add at least one reference image."); return; }

  const form = new FormData();
  form.append("name", name);
  form.append("person_id", document.getElementById("f-id").value.trim());
  form.append("description", document.getElementById("f-desc").value.trim());
  stagedFiles.forEach((f) => form.append("images", f, f.name));

  const btn = document.getElementById("enroll-submit");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> Enrolling…`;
  try {
    const res = await api(`/api/people`, { method: "POST", body: form });
    toast(`${res.message}. ${res.images_skipped?.length ? `${res.images_skipped.length} image(s) skipped.` : ""}`);
    location.hash = "#/people";
  } catch (err) {
    toast(err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = "✚ Enroll person";
  }
}

/* ---------------------------------------------------------------- PEOPLE */

async function renderPeople(content) {
  content.innerHTML = `
    <div class="eyebrow">People directory</div>
    <h1 class="page-title">Enrolled identities</h1>
    <p class="page-desc">Everyone currently enrolled in the server-side SQLite database.</p>
    <div id="people-grid" class="people-grid"><div class="empty">Loading…</div></div>
  `;
  const { people } = await api("/api/people");
  const grid = document.getElementById("people-grid");
  if (!people.length) {
    grid.innerHTML = `<div class="empty">No one enrolled yet. Go to <a href="#/enroll" style="color:var(--accent)">Enroll identity</a> to add your first person.</div>`;
    return;
  }
  grid.innerHTML = people.map((p) => `
    <div class="person-card">
      <div class="person-avatar">${escapeHtml(p.name.slice(0, 1).toUpperCase())}</div>
      <div class="person-name">${escapeHtml(p.name)}</div>
      <div class="person-meta">${p.person_id ? escapeHtml(p.person_id) + " · " : ""}${p.num_images} reference image${p.num_images === 1 ? "" : "s"}</div>
      <div class="person-desc">${escapeHtml(p.description || "No description provided.")}</div>
      <div class="person-actions">
        <button class="btn btn-danger btn-sm" data-name="${escapeHtml(p.name)}">Remove</button>
      </div>
    </div>
  `).join("");

  grid.querySelectorAll("button[data-name]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      if (!confirm(`Remove ${btn.dataset.name} from the directory?`)) return;
      await api(`/api/people/${encodeURIComponent(btn.dataset.name)}`, { method: "DELETE" });
      toast(`Removed ${btn.dataset.name}`);
      renderPeople(content);
    });
  });
}

/* ---------------------------------------------------------------- EVALUATION */

async function renderEvaluation(content) {
  let status = null;
  try { status = await api("/api/evaluation/status"); } catch (err) { status = { database_samples: 0, known_samples: 0, unknown_samples: 0, people: 0, message: err.message }; }
  content.innerHTML = `
    <div class="eyebrow">Evaluation lab / SQLite validation</div>
    <h1 class="page-title">Measure the decision boundary</h1>
    <p class="page-desc">Every run reads the <b>current</b> SQLite database. Enrollment, deletion, and demo-data changes therefore change the evaluation population and results automatically.</p>

    <div class="grid grid-4" style="margin-bottom:16px">
      ${statCard("◫", status.database_samples, "Database samples", "current SQLite set")}
      ${statCard("👥", status.people, "Enrolled identities", "current people table")}
      ${statCard("✔", status.known_samples, "Known samples", "leave-one-out eligible")}
      ${statCard("⚠", status.unknown_samples, "Unknown samples", "rejection tests")}
    </div>

    <div class="card" style="margin-bottom:16px">
      <div class="section-title">Evaluation controls</div>
      <div class="section-sub">No dataset folder is required. Images and embeddings are read directly from <code>face_recognition.db</code>.</div>
      <div style="display:flex;gap:10px;flex-wrap:wrap">
        <button class="btn btn-primary" id="run-eval">▶ Run evaluation</button>
        <button class="btn" id="load-demo">＋ Load 10 demo identities</button>
      </div>
      <div id="eval-status" class="section-sub" style="margin-top:12px">${escapeHtml(status.message || "Ready")}</div>
    </div>
    <div id="eval-results"></div>
  `;
  document.getElementById("run-eval").addEventListener("click", runEvaluationNow);
  document.getElementById("load-demo").addEventListener("click", async () => {
    const btn = document.getElementById("load-demo");
    btn.disabled = true; btn.innerHTML = `<span class="spinner"></span> Loading demo data…`;
    try {
      const result = await api("/api/demo/load", { method: "POST" });
      toast(`Demo load finished: ${result.added.length} added.`);
      renderEvaluation(content);
    } catch (err) {
      toast(err.message);
    } finally {
      btn.disabled = false; btn.innerHTML = "＋ Load 10 demo identities";
    }
  });
}

async function runEvaluationNow() {
  const btn = document.getElementById("run-eval");
  const results = document.getElementById("eval-results");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> Evaluating…`;
  results.innerHTML = `<div class="empty">Running MTCNN + embedding + matching over the current SQLite samples…</div>`;
  try {
    const summary = await api("/api/evaluate", { method: "POST" });
    renderEvalSummary(summary);
  } catch (err) {
    results.innerHTML = `<div class="empty">${escapeHtml(err.message)}</div>`;
  } finally {
    btn.disabled = false;
    btn.innerHTML = "▶ Run evaluation";
  }
}

function metricBar(label, value, max=100, klass="") {
  const v = Math.max(0, Math.min(max, Number(value) || 0));
  return `<div class="kv-row"><span>${escapeHtml(label)}</span><b>${Number(value).toFixed(2)}%</b></div><div style="height:9px;border-radius:999px;background:var(--border-soft);overflow:hidden;margin:4px 0 14px"><div style="height:100%;width:${(v/max)*100}%;background:linear-gradient(90deg,var(--accent),var(--accent-2));border-radius:999px"></div></div>`;
}

function renderEvalSummary(summary) {
  const results = document.getElementById("eval-results");
  if (!summary.available) {
    results.innerHTML = `<div class="empty">${escapeHtml(summary.message || "No evaluation samples are currently stored in SQLite.")}</div>`;
    return;
  }
  const rows = (summary.rows || []).slice(0, 30).map((r) => `
    <tr>
      <td>${escapeHtml(r.image || "")}</td>
      <td>${escapeHtml(r.expected || "")}</td>
      <td>${escapeHtml(r.predicted || "")}</td>
      <td>${Number(r.similarity || 0).toFixed(3)}</td>
      <td><span class="badge ${r.result === "correct" ? "badge-known" : "badge-unknown"}">${escapeHtml(r.result || "")}</span></td>
    </tr>
  `).join("");

  const perClass = Object.entries(summary.per_class || {}).map(([name, s]) => `
    <div style="margin-bottom:12px">
      <div class="kv-row"><span>${escapeHtml(name)}</span><b>${s.correct}/${s.total} · ${Number(s.accuracy).toFixed(1)}%</b></div>
      <div style="height:7px;border-radius:999px;background:var(--border-soft);overflow:hidden"><div style="height:100%;width:${Math.max(0,Math.min(100,Number(s.accuracy)||0))}%;background:var(--good)"></div></div>
    </div>
  `).join("");

  const chart = `
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;align-items:end;height:210px;padding:16px;border:1px solid var(--border-soft);border-radius:12px;background:var(--card-2)">
      ${[
        ["Accuracy", summary.accuracy, "var(--good)"],
        ["False accept", summary.false_accept_rate, "var(--bad)"],
        ["False reject", summary.false_reject_rate, "var(--warn)"],
        ["Detection fail", summary.detection_failure_rate, "var(--accent)"]
      ].map(([label,value,color]) => `<div style="height:100%;display:flex;flex-direction:column;justify-content:end;align-items:center;gap:8px"><b>${Number(value).toFixed(1)}%</b><div style="width:70%;height:${Math.max(4,Math.min(100,Number(value)||0))}%;background:${color};border-radius:8px 8px 2px 2px;min-height:4px"></div><span style="font-size:11px;color:var(--text-dim);text-align:center">${label}</span></div>`).join("")}
    </div>`;

  results.innerHTML = `
    <div class="metric-4">
      ${statCard("🎯", summary.accuracy + "%", "Accuracy", summary.evaluated_samples + " evaluated")}
      ${statCard("⚠", summary.false_accept_rate + "%", "False accept rate", summary.evaluated_unknown + " unknown")}
      ${statCard("↩", summary.false_reject_rate + "%", "False reject rate", summary.evaluated_known + " known")}
      ${statCard("◫", summary.detection_failure_rate + "%", "Detection failures", summary.detection_failures + " failed")}
    </div>
    <div class="two-col" style="margin-top:16px">
      <div class="card">
        <div class="section-title">Evaluation metrics</div>
        <div class="section-sub">Threshold: ${Number(summary.threshold).toFixed(2)} · leave-one-out for known identities</div>
        ${chart}
      </div>
      <div class="card">
        <div class="section-title">Per-identity breakdown</div>
        <div class="section-sub">Only evaluable known samples contribute to these values.</div>
        ${perClass || '<div class="empty">No multi-reference identities are available yet.</div>'}
      </div>
    </div>
    <div class="card" style="margin-top:16px">
      <div class="section-title">Sample-level results</div>
      <div class="section-sub">Run ID ${summary.run_id || "—"} · ${summary.samples} samples in SQLite · ${summary.skipped_single_reference} single-reference samples skipped from known accuracy.</div>
      <div style="overflow:auto;max-height:420px"><table><thead><tr><th>Image</th><th>Expected</th><th>Predicted</th><th>Similarity</th><th>Result</th></tr></thead><tbody>${rows}</tbody></table></div>
    </div>
  `;
}

/* ---------------------------------------------------------------- MODEL NOTES */

async function renderNotes(content) {
  const info = await api("/api/model-info");
  content.innerHTML = `
    <div class="eyebrow">Model notes / Explainability</div>
    <h1 class="page-title">From pixels to decision</h1>
    <p class="page-desc">VisionID keeps detection, representation, storage, matching, and
    rejection separate so operators can inspect the complete signal path.</p>

    <div class="two-col">
      <div class="card">
        <div class="section-title">Recognition pipeline</div>
        ${info.pipeline.map((step, i) => `
          <div class="pipeline-step">
            <div class="pipeline-num">${i + 1}</div>
            <div>
              <div class="pipeline-title">${escapeHtml(step.step)}</div>
              <div class="pipeline-detail">${escapeHtml(step.detail)}</div>
            </div>
          </div>
        `).join("")}
      </div>

      <div class="card">
        <div class="section-title">Runtime profile</div>
        <div class="kv-row"><span>Recognition model</span><b>${escapeHtml(info.runtime.recognition_model)}</b></div>
        <div class="kv-row"><span>Detector</span><b>${escapeHtml(info.runtime.detector)}</b></div>
        <div class="kv-row"><span>Embedding dimension</span><b>${info.runtime.embedding_dimension}</b></div>
        <div class="kv-row"><span>Execution provider</span><b>${escapeHtml(info.runtime.execution_provider)}</b></div>
        <div class="kv-row"><span>Similarity metric</span><b>${escapeHtml(info.runtime.similarity_metric)}</b></div>
        <div class="kv-row"><span>Current threshold</span><b>${Math.round(info.runtime.threshold * 100)}%</b></div>
        <div class="note-box" style="margin-top:16px">${escapeHtml(info.limitations)}</div>
      </div>
    </div>
  `;
}
