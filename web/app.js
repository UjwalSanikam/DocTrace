"use strict";
const el = id => document.getElementById(id);
const state = {
  file: null, result: null, original: null, probability: null,
  alternative: null, worst_case: null, ela: null,
  view: "overlay", busy: false, ready: false
};
const canvas = el("canvas"), context = canvas.getContext("2d", {willReadFrequently: true});
const tabs = [...document.querySelectorAll("[data-view]")];
const pct = number => `${(number * 100).toFixed(2)}%`;

function message(text, error = false) { el("message").textContent = text; el("message").classList.toggle("error", error); }
function setBusy(busy) {
  state.busy = busy;
  el("busy").hidden = !busy;
  ["file", "sample", "probe"].forEach(id => el(id).disabled = busy);
  el("analyze").disabled = busy || !state.file || !state.ready;
  el("analyze").firstChild.textContent = busy ? "Analyzing… " : "Analyze document ";
}
function enableResults(enabled) {
  ["threshold", "opacity", "save-mask", "save-overlay", "save-ela", "save-report"].forEach(id => el(id).disabled = !enabled);
  tabs.find(t => t.dataset.view === "sensitivity").disabled = !enabled || !state.alternative;
  tabs.find(t => t.dataset.view === "worst_case").disabled = !enabled || !state.worst_case;
  tabs.find(t => t.dataset.view === "ela").disabled = !enabled || !state.ela;
}
async function loadPixels(url) {
  const image = new Image();
  image.src = url;
  await image.decode();
  if (image.naturalWidth * image.naturalHeight > 8000000 || Math.max(image.naturalWidth, image.naturalHeight) > 8192) throw new Error("Choose an image up to 8 megapixels and 8192 pixels per side.");
  const buffer = document.createElement("canvas");
  buffer.width = image.naturalWidth; buffer.height = image.naturalHeight;
  const ctx = buffer.getContext("2d", {willReadFrequently: true});
  ctx.fillStyle = "white"; ctx.fillRect(0, 0, buffer.width, buffer.height); ctx.drawImage(image, 0, 0);
  return ctx.getImageData(0, 0, buffer.width, buffer.height);
}
function chooseView(view) {
  if (view === "sensitivity" && !state.alternative) return;
  if (view === "worst_case" && !state.worst_case) return;
  if (view === "ela" && !state.ela) return;
  state.view = view;
  tabs.forEach(tab => tab.setAttribute("aria-selected", String(tab.dataset.view === view)));
  render();
}
function measures() {
  let positive = 0, flips = 0;
  const threshold = Number(el("threshold").value) / 100;
  const cutoff = Math.ceil(threshold * 255);
  const count = state.probability.width * state.probability.height;
  for (let i = 0; i < count * 4; i += 4) {
    const active = state.probability.data[i] >= cutoff;
    positive += Number(active);
    // worst_case flips: any pixel whose probability changes by >= 10% under any probe.
    if (state.worst_case) flips += Number(state.worst_case.data[i] >= Math.ceil(0.1 * 255));
  }
  return {threshold, flagged_pixel_fraction: positive / count, worst_case_flip_fraction: state.worst_case ? flips / count : null};
}
function draw(view) {
  const source = state.original;
  const result = new ImageData(new Uint8ClampedArray(source.data), source.width, source.height);
  if (!state.probability || view === "original") return result;
  const cutoff = Math.ceil(Number(el("threshold").value) / 100 * 255);
  const alpha = Number(el("opacity").value) / 100;
  for (let i = 0; i < result.data.length; i += 4) {
    const p = state.probability.data[i] / 255;
    if (view === "mask") {
      result.data[i] = result.data[i+1] = result.data[i+2] = state.probability.data[i] >= cutoff ? 255 : 0;
    } else if (view === "overlay" && state.probability.data[i] >= cutoff) {
      [242, 90, 62].forEach((color, k) => result.data[i+k] = source.data[i+k] * (1-alpha) + color * alpha);
    } else if (view === "probability") {
      const color = [25, 88, 71];
      color.forEach((channel, k) => result.data[i+k] = 250 * (1-p) + channel * p);
    } else if (view === "sensitivity" && state.alternative) {
      const amount = Math.min(1, Math.abs(p - state.alternative.data[i]/255) / 0.25);
      [220, 106, 62].forEach((channel, k) => result.data[i+k] = 250 * (1-amount) + channel * amount);
    } else if (view === "worst_case" && state.worst_case) {
      // Show worst-case sensitivity: white → deep orange.
      const amount = Math.min(1, state.worst_case.data[i] / 255 / 0.25);
      [180, 60, 20].forEach((channel, k) => result.data[i+k] = 250 * (1-amount) + channel * amount);
    } else if (view === "ela" && state.ela) {
      // ELA: map per-channel residual to a purple-tinted heatmap for contrast with NN overlay.
      const rEla = state.ela.data[i] / 255;
      const gEla = state.ela.data[i+1] / 255;
      const bEla = state.ela.data[i+2] / 255;
      const magnitude = Math.min(1, (rEla + gEla + bEla) / 3 * 3);
      result.data[i]   = Math.round(250 * (1-magnitude) + 160 * magnitude);
      result.data[i+1] = Math.round(250 * (1-magnitude) + 30  * magnitude);
      result.data[i+2] = Math.round(250 * (1-magnitude) + 200 * magnitude);
    }
  }
  return result;
}
function renderRegions(regions) {
  const card = el("regions-card");
  const list = el("regions-list");
  list.innerHTML = "";
  if (!regions || regions.length === 0) { card.hidden = true; return; }
  card.hidden = false;
  regions.forEach(r => {
    const item = document.createElement("div");
    item.className = "region-item";
    item.innerHTML = `<span class="region-rank">#${r.rank}</span>
      <div class="region-body">
        <strong>${(r.area_fraction * 100).toFixed(2)}%</strong> of image
        &nbsp;·&nbsp; confidence <strong>${(r.mean_confidence * 100).toFixed(1)}%</strong>
        <div class="region-bbox">bbox (y,x): ${r.bbox[0]},${r.bbox[1]} → ${r.bbox[2]},${r.bbox[3]}</div>
      </div>`;
    list.appendChild(item);
  });
}
function renderMultiProbe(multi_probe) {
  // Update the flips stat label to show worst-case summary.
  if (multi_probe) {
    el("flips").textContent = pct(multi_probe.worst_case_flip_fraction);
  }
}
function render() {
  if (!state.original) return;
  canvas.width = state.original.width; canvas.height = state.original.height;
  context.putImageData(draw(state.view), 0, 0);
  canvas.hidden = false; el("empty").hidden = true;
  el("dimensions").textContent = `${canvas.width} × ${canvas.height}`;
  el("threshold-value").textContent = (Number(el("threshold").value)/100).toFixed(2);
  el("opacity-value").textContent = `${el("opacity").value}%`;
  const legends = {
    overlay: "Coral marks pixels above the selected threshold",
    probability: "White → green: model response from 0 to 1",
    sensitivity: "White → coral: JPEG-70 absolute response change (0 to 0.25+)",
    worst_case: "White → orange: worst-case change across all 4 probe conditions",
    ela: "White → purple: Error Level Analysis residual (amplified ×15)",
    original: "Original image · no overlay",
  };
  el("legend-text").textContent = state.result ? legends[state.view] : "Image loaded · ready for analysis";
  el("legend-color").style.background = {
    overlay: "#e38160", probability: "#195847", sensitivity: "#e38160",
    worst_case: "#b43c14", ela: "#a01ec8", original: "#c9cec8",
  }[state.view] || "#e38160";
  if (state.probability) {
    const metric = measures();
    el("area").textContent = pct(metric.flagged_pixel_fraction);
    el("flips").textContent = metric.worst_case_flip_fraction !== null ? pct(metric.worst_case_flip_fraction) : "Not checked";
  }
}
async function selectFile(file) {
  if (!file || state.busy) return;
  try {
    if (file.size > 24*1024*1024) throw new Error("Choose an image smaller than 24 MB.");
    if (!["image/png", "image/jpeg", "image/webp"].includes(file.type)) throw new Error("Choose a PNG, JPEG, or WebP image.");
    const url = URL.createObjectURL(file);
    let pixels;
    try { pixels = await loadPixels(url); } finally { URL.revokeObjectURL(url); }
    state.file = file; state.original = pixels; state.result = state.probability = state.alternative = state.worst_case = state.ela = null;
    enableResults(false); chooseView("overlay"); setBusy(false);
    el("file-info").textContent = file.name;
    ["area", "flips", "tiles"].forEach(id => el(id).textContent = "—");
    el("elapsed").textContent = "Ready for analysis";
    el("regions-card").hidden = true;
    message(file.name === "synthetic-sample.png" ? "This sample is procedurally generated and unedited. Any highlights are false positives; use it to explore the interface." : "Image loaded. Run analysis to inspect model responses.");
  } catch (error) { message(error.message, true); }
}
el("file").addEventListener("change", event => selectFile(event.target.files[0]));
el("sample").addEventListener("click", async () => {
  try {
    const response = await fetch("/api/sample");
    if (!response.ok) throw new Error("Sample could not be loaded.");
    await selectFile(new File([await response.blob()], "synthetic-sample.png", {type:"image/png"}));
  } catch (error) { message(error.message, true); }
});
["dragover", "dragleave", "drop"].forEach(name => el("drop").addEventListener(name, event => {
  event.preventDefault();
  el("drop").classList.toggle("drag", name === "dragover");
  if (name === "drop") selectFile(event.dataTransfer.files[0]);
}));
el("analyze").addEventListener("click", async () => {
  if (!state.file || state.busy) return;
  setBusy(true); enableResults(false); message("Analyzing locally. Running neural prediction, ELA, and multi-probe sensitivity…");
  try {
    const response = await fetch(`/api/analyze?probe=${el("probe").checked ? 1 : 0}`, {method:"POST", headers:{"Content-Type":state.file.type, "X-DocTrace-Request":"1"}, body:state.file});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Analysis failed.");
    const url = content => `data:image/png;base64,${content}`;
    const [original, probability, alternative, worst_case, elaPixels] = await Promise.all([
      loadPixels(url(data.image)),
      loadPixels(url(data.probability)),
      data.alternative ? loadPixels(url(data.alternative)) : Promise.resolve(null),
      data.worst_case  ? loadPixels(url(data.worst_case))  : Promise.resolve(null),
      data.ela         ? loadPixels(url(data.ela))          : Promise.resolve(null),
    ]);
    state.original = original; state.probability = probability;
    state.alternative = alternative; state.worst_case = worst_case; state.ela = elaPixels;
    state.result = data.report;
    el("tiles").textContent = data.report.tiles;
    el("elapsed").textContent = `${data.report.elapsed_seconds.toFixed(2)} s · ${data.report.device.toUpperCase()}`;
    chooseView("overlay"); enableResults(true);
    renderRegions(data.report.regions);
    if (data.report.multi_probe) renderMultiProbe(data.report.multi_probe);
    message(state.file.name === "synthetic-sample.png" ? "Analysis complete. This is an unedited synthetic sample: highlighted areas are false positives." : "Analysis complete. Inspect the overlay, ELA, and worst-case sensitivity views before interpreting any highlighted region.");
  } catch (error) { message(error.message, true); enableResults(Boolean(state.result)); }
  finally { setBusy(false); }
});
tabs.forEach(tab => tab.addEventListener("click", () => chooseView(tab.dataset.view)));
["threshold", "opacity"].forEach(id => el(id).addEventListener("input", render));
function download(blob, suffix) {
  const url = URL.createObjectURL(blob), anchor = document.createElement("a");
  const name = state.file.name.replace(/\.[^.]*$/, "").replace(/[^a-zA-Z0-9_-]/g, "_") || "document";
  anchor.href = url; anchor.download = `${name}-${suffix}`; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function saveImage(view) {
  if (!state.result) return;
  const out = document.createElement("canvas"); out.width = canvas.width; out.height = canvas.height;
  out.getContext("2d").putImageData(draw(view), 0, 0);
  out.toBlob(blob => {if (blob) download(blob, `${view}.png`);}, "image/png");
}
el("save-overlay").addEventListener("click", () => saveImage("overlay"));
el("save-mask").addEventListener("click", () => saveImage("mask"));
el("save-ela").addEventListener("click", () => { if (state.ela) saveImage("ela"); });
el("save-report").addEventListener("click", () => {
  if (!state.result) return;
  const values = measures();
  const report = {
    ...state.result,
    input_name: state.file.name,
    threshold: values.threshold,
    flagged_pixel_fraction: values.flagged_pixel_fraction,
    display_note: "Interactive masks and fractions use 8-bit probability maps (rounding error up to 0.5/255); benchmark metrics use full precision.",
    probe: state.result.probe ? {...state.result.probe, threshold_flip_fraction: state.result.probe.threshold_flip_fraction} : null,
  };
  download(new Blob([JSON.stringify(report, null, 2)], {type:"application/json"}), "report.json");
});
fetch("/api/status").then(async response => {
  if (!response.ok) throw new Error("Model unavailable");
  const status = await response.json(); state.ready = true;
  el("connection").textContent = `${status.device === "cuda" ? "GPU" : "CPU"} ready · local processing`;
  el("model-meta").textContent = `${status.model} · epoch ${status.epoch} · checkpoint ${status.checkpoint_sha256.slice(0, 12)}`;
  setBusy(false);
}).catch(() => {el("connection").textContent = "Model offline"; message("The model server is unavailable. Restart the app from the project terminal.", true);});
