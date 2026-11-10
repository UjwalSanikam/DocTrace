"use strict";

const el = id => document.getElementById(id);
const state = {
  file: null,
  result: null,          // Server analysis report at analysis_threshold (0.50)
  original: null,        // ImageData RGBA
  probability: null,     // ImageData RGBA (8-bit quantized: R=channel=prob*255)
  alternative: null,     // ImageData RGBA (8-bit jpeg70)
  worst_case: null,      // ImageData RGBA (8-bit worst-case abs change)
  ela_residual: null,    // ImageData RGBA (amplified JPEG residual)
  view: "overlay",
  busy: false,
  ready: false,
  selectedRegionIndex: null,
  interactiveRegions: [],
  interactiveStats: null
};

const canvas = el("canvas");
const context = canvas.getContext("2d", { willReadFrequently: true });
const tabs = [...document.querySelectorAll(".tabs [data-view]")];
const pct = number => `${(number * 100).toFixed(2)}%`;

function message(text, error = false) {
  const m = el("message");
  m.textContent = text;
  m.classList.toggle("error", error);
}

function setBusy(busy) {
  state.busy = busy;
  el("busy").hidden = !busy;
  ["file", "sample", "probe"].forEach(id => el(id).disabled = busy);
  el("analyze").disabled = busy || !state.file || !state.ready;
  el("analyze").firstChild.textContent = busy ? "Analyzing… " : "Analyze document ";
}

function enableResults(enabled) {
  const hasResult = enabled && Boolean(state.result);
  const hasAlternative = hasResult && Boolean(state.alternative);
  const hasWorst = hasResult && Boolean(state.worst_case);
  const hasEla = hasResult && Boolean(state.ela_residual);

  ["threshold", "opacity", "save-mask", "save-overlay", "save-report"].forEach(id => {
    el(id).disabled = !hasResult;
  });

  el("save-ela-residual").disabled = !hasEla;
  el("save-ela-rendered").disabled = !hasEla;
  el("save-sensitivity-worst").disabled = !hasWorst;
  el("save-sensitivity-jpeg70").disabled = !hasAlternative;

  tabs.find(t => t.dataset.view === "sensitivity").disabled = !hasAlternative;
  tabs.find(t => t.dataset.view === "worst_case").disabled = !hasWorst;
  tabs.find(t => t.dataset.view === "ela").disabled = !hasEla;
}

async function loadPixels(url) {
  const image = new Image();
  image.src = url;
  await image.decode();
  if (image.naturalWidth * image.naturalHeight > 8000000 || Math.max(image.naturalWidth, image.naturalHeight) > 8192) {
    throw new Error("Choose an image up to 8 megapixels and 8192 pixels per side.");
  }
  const buffer = document.createElement("canvas");
  buffer.width = image.naturalWidth;
  buffer.height = image.naturalHeight;
  const ctx = buffer.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, buffer.width, buffer.height);
  ctx.drawImage(image, 0, 0);
  return ctx.getImageData(0, 0, buffer.width, buffer.height);
}

function chooseView(view) {
  if (view === "sensitivity" && !state.alternative) return;
  if (view === "worst_case" && !state.worst_case) return;
  if (view === "ela" && !state.ela_residual) return;
  state.view = view;
  tabs.forEach(tab => {
    const isSelected = tab.dataset.view === view;
    tab.setAttribute("aria-selected", String(isSelected));
    tab.setAttribute("tabindex", isSelected ? "0" : "-1");
  });
  render();
}

/**
 * Fast 4-connectivity connected-component extraction on quantized 8-bit probability map.
 */
function extractRegions(probData, width, height, threshold, maxRegions = 10) {
  const cutoff = Math.ceil(threshold * 255);
  const totalPixels = width * height;
  const labels = new Int32Array(totalPixels);
  const parent = [0];
  let nextLabel = 1;

  for (let y = 0; y < height; y++) {
    const rowOffset = y * width;
    for (let x = 0; x < width; x++) {
      const idx = rowOffset + x;
      // Probability stored in red channel (idx * 4)
      if (probData[idx * 4] < cutoff) continue;

      const up = y > 0 ? labels[idx - width] : 0;
      const left = x > 0 ? labels[idx - 1] : 0;

      if (up === 0 && left === 0) {
        parent.push(nextLabel);
        labels[idx] = nextLabel;
        nextLabel++;
      } else if (up === 0) {
        let r = left;
        while (parent[r] !== r) r = parent[r];
        labels[idx] = r;
      } else if (left === 0) {
        let r = up;
        while (parent[r] !== r) r = parent[r];
        labels[idx] = r;
      } else {
        let ru = up;
        while (parent[ru] !== ru) ru = parent[ru];
        let rl = left;
        while (parent[rl] !== rl) rl = parent[rl];
        labels[idx] = ru;
        if (ru !== rl) {
          parent[rl] = ru;
        }
      }
    }
  }

  if (nextLabel === 1) {
    return { total_count: 0, returned_count: 0, truncated: false, threshold, regions: [] };
  }

  // Flatten parent pointers
  for (let i = 1; i < parent.length; i++) {
    let r = i;
    while (parent[r] !== r) r = parent[r];
    parent[i] = r;
  }

  // Remap roots to 1..numComponents
  const rootMap = new Map();
  let numComponents = 0;
  for (let i = 1; i < parent.length; i++) {
    const r = parent[i];
    if (!rootMap.has(r)) {
      numComponents++;
      rootMap.set(r, numComponents);
    }
  }

  // Aggregate stats in a single pass
  const areas = new Uint32Array(numComponents + 1);
  const sums = new Float64Array(numComponents + 1);
  const yMin = new Int32Array(numComponents + 1).fill(height);
  const yMax = new Int32Array(numComponents + 1).fill(-1);
  const xMin = new Int32Array(numComponents + 1).fill(width);
  const xMax = new Int32Array(numComponents + 1).fill(-1);

  for (let y = 0; y < height; y++) {
    const rowOffset = y * width;
    for (let x = 0; x < width; x++) {
      const idx = rowOffset + x;
      const rawLab = labels[idx];
      if (rawLab === 0) continue;
      const cid = rootMap.get(parent[rawLab]);
      areas[cid]++;
      sums[cid] += probData[idx * 4] / 255;
      if (y < yMin[cid]) yMin[cid] = y;
      if (y > yMax[cid]) yMax[cid] = y;
      if (x < xMin[cid]) xMin[cid] = x;
      if (x > xMax[cid]) xMax[cid] = x;
    }
  }

  // Deterministic sort: area desc, mean response desc, yMin asc, xMin asc
  const indices = [];
  for (let c = 1; c <= numComponents; c++) {
    indices.push(c);
  }
  indices.sort((a, b) => {
    if (areas[b] !== areas[a]) return areas[b] - areas[a];
    const meanB = sums[b] / areas[b];
    const meanA = sums[a] / areas[a];
    if (meanB !== meanA) return meanB - meanA;
    if (yMin[a] !== yMin[b]) return yMin[a] - yMin[b];
    if (xMin[a] !== xMin[b]) return xMin[a] - xMin[b];
    if (yMax[a] !== yMax[b]) return yMax[a] - yMax[b];
    return xMax[a] - xMax[b];
  });

  const limit = maxRegions === null ? numComponents : Math.min(maxRegions, numComponents);
  const regions = [];
  for (let rank = 1; rank <= limit; rank++) {
    const cid = indices[rank - 1];
    const area = areas[cid];
    const meanResp = Number((sums[cid] / area).toFixed(4));
    regions.push({
      rank,
      area_px: area,
      area_fraction: Number((area / totalPixels).toFixed(6)),
      mean_model_response: meanResp,
      mean_confidence: meanResp, // backward compatibility
      bbox: [yMin[cid], xMin[cid], yMax[cid], xMax[cid]]
    });
  }

  return {
    total_count: numComponents,
    returned_count: regions.length,
    truncated: numComponents > regions.length,
    threshold,
    regions
  };
}

function calculateMeasures(threshold) {
  const cutoff = Math.ceil(threshold * 255);
  const count = state.probability.width * state.probability.height;
  let positive = 0;
  let jpeg70Flips = 0;
  let largeResponseChanges = 0;
  const changeCutoff = Math.ceil(0.10 * 255);

  const probData = state.probability.data;
  const altData = state.alternative ? state.alternative.data : null;
  const worstData = state.worst_case ? state.worst_case.data : null;

  for (let i = 0; i < count * 4; i += 4) {
    const p = probData[i];
    const active = p >= cutoff;
    if (active) positive++;

    if (altData) {
      const altActive = altData[i] >= cutoff;
      if (active !== altActive) jpeg70Flips++;
    }

    if (worstData && worstData[i] >= changeCutoff) {
      largeResponseChanges++;
    }
  }

  return {
    threshold,
    flagged_pixel_fraction: positive / count,
    jpeg70_threshold_flip_fraction: altData ? jpeg70Flips / count : null,
    worst_case_large_response_change_fraction: worstData ? largeResponseChanges / count : null,
    response_change_cutoff: 0.10
  };
}

function draw(view, threshold) {
  const source = state.original;
  const result = new ImageData(new Uint8ClampedArray(source.data), source.width, source.height);
  if (!state.probability || view === "original") return result;

  const cutoff = Math.ceil(threshold * 255);
  const alpha = Number(el("opacity").value) / 100;
  const probData = state.probability.data;
  const altData = state.alternative ? state.alternative.data : null;
  const worstData = state.worst_case ? state.worst_case.data : null;
  const elaData = state.ela_residual ? state.ela_residual.data : null;

  for (let i = 0; i < result.data.length; i += 4) {
    const p = probData[i] / 255;
    if (view === "mask") {
      const v = probData[i] >= cutoff ? 255 : 0;
      result.data[i] = result.data[i + 1] = result.data[i + 2] = v;
    } else if (view === "overlay" && probData[i] >= cutoff) {
      result.data[i]     = source.data[i]     * (1 - alpha) + 242 * alpha;
      result.data[i + 1] = source.data[i + 1] * (1 - alpha) + 90  * alpha;
      result.data[i + 2] = source.data[i + 2] * (1 - alpha) + 62  * alpha;
    } else if (view === "probability") {
      result.data[i]     = 250 * (1 - p) + 25 * p;
      result.data[i + 1] = 250 * (1 - p) + 88 * p;
      result.data[i + 2] = 250 * (1 - p) + 71 * p;
    } else if (view === "sensitivity" && altData) {
      const amount = Math.min(1, Math.abs(p - altData[i] / 255) / 0.25);
      result.data[i]     = 250 * (1 - amount) + 220 * amount;
      result.data[i + 1] = 250 * (1 - amount) + 106 * amount;
      result.data[i + 2] = 250 * (1 - amount) + 62  * amount;
    } else if (view === "worst_case" && worstData) {
      const amount = Math.min(1, (worstData[i] / 255) / 0.25);
      result.data[i]     = 250 * (1 - amount) + 180 * amount;
      result.data[i + 1] = 250 * (1 - amount) + 60  * amount;
      result.data[i + 2] = 250 * (1 - amount) + 20  * amount;
    } else if (view === "ela" && elaData) {
      // Colorized inspection view for ELA residual
      const r = elaData[i] / 255;
      const g = elaData[i + 1] / 255;
      const b = elaData[i + 2] / 255;
      const magnitude = Math.min(1, ((r + g + b) / 3) * 3);
      result.data[i]     = Math.round(250 * (1 - magnitude) + 160 * magnitude);
      result.data[i + 1] = Math.round(250 * (1 - magnitude) + 30  * magnitude);
      result.data[i + 2] = Math.round(250 * (1 - magnitude) + 200 * magnitude);
    }
  }
  return result;
}

function drawBoundingBox(bbox) {
  if (!bbox || bbox.length !== 4) return;
  const [y0, x0, y1, x1] = bbox;
  const w = Math.max(1, x1 - x0 + 1);
  const h = Math.max(1, y1 - y0 + 1);

  context.save();
  // Semi-transparent highlight fill
  context.fillStyle = "rgba(215, 107, 72, 0.18)";
  context.fillRect(x0, y0, w, h);

  // Outer contrast stroke
  context.lineWidth = 2;
  context.strokeStyle = "#ffffff";
  context.strokeRect(x0 - 0.5, y0 - 0.5, w + 1, h + 1);

  // Main accent stroke
  context.strokeStyle = "#d76b48";
  context.strokeRect(x0, y0, w, h);

  // Corner markers for crisp identification
  context.fillStyle = "#203e39";
  const markerSize = Math.min(6, Math.max(2, Math.floor(Math.min(w, h) / 3)));
  context.fillRect(x0 - markerSize / 2, y0 - markerSize / 2, markerSize, markerSize);
  context.fillRect(x0 + w - markerSize / 2, y0 - markerSize / 2, markerSize, markerSize);
  context.fillRect(x0 - markerSize / 2, y0 + h - markerSize / 2, markerSize, markerSize);
  context.fillRect(x0 + w - markerSize / 2, y0 + h - markerSize / 2, markerSize, markerSize);
  context.restore();
}

function renderRegionsUI(regionExtraction) {
  const card = el("regions-card");
  const list = el("regions-list");
  const note = el("regions-note");
  list.innerHTML = "";

  if (!regionExtraction || regionExtraction.total_count === 0) {
    card.hidden = false;
    list.innerHTML = `<div class="empty-regions">No suspicious regions exceed display threshold ${(regionExtraction.threshold).toFixed(2)}.<small>Lower the threshold slider to inspect smaller response areas.</small></div>`;
    note.textContent = "0 regions found above current display threshold.";
    return;
  }

  card.hidden = false;
  const countLabel = regionExtraction.truncated
    ? `Showing top ${regionExtraction.returned_count} of ${regionExtraction.total_count} regions (display threshold ${(regionExtraction.threshold).toFixed(2)}).`
    : `${regionExtraction.total_count} region(s) at display threshold ${(regionExtraction.threshold).toFixed(2)}.`;
  note.textContent = `${countLabel} Click a region to highlight its bounding box on the image.`;

  regionExtraction.regions.forEach((r, idx) => {
    const item = document.createElement("div");
    item.className = "region-item" + (state.selectedRegionIndex === idx ? " selected" : "");
    item.setAttribute("role", "button");
    item.setAttribute("tabindex", "0");
    item.setAttribute("aria-label", `Region ${r.rank}, ${(r.area_fraction * 100).toFixed(2)} percent area`);

    item.innerHTML = `
      <span class="region-rank">#${r.rank}</span>
      <div class="region-body">
        <strong>${(r.area_fraction * 100).toFixed(2)}%</strong> of image (${r.area_px.toLocaleString()} px)
        &nbsp;·&nbsp; mean response <strong>${(r.mean_model_response * 100).toFixed(1)}%</strong>
        <div class="region-bbox">bbox (y,x): ${r.bbox[0]},${r.bbox[1]} → ${r.bbox[2]},${r.bbox[3]}</div>
      </div>
    `;

    const toggleSelect = () => {
      state.selectedRegionIndex = state.selectedRegionIndex === idx ? null : idx;
      render();
    };

    item.addEventListener("click", toggleSelect);
    item.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        toggleSelect();
      }
    });

    list.appendChild(item);
  });
}

function render() {
  if (!state.original) return;
  const displayThreshold = Number(el("threshold").value) / 100;
  el("threshold-value").textContent = displayThreshold.toFixed(2);
  el("opacity-value").textContent = `${el("opacity").value}%`;

  canvas.width = state.original.width;
  canvas.height = state.original.height;
  context.putImageData(draw(state.view, displayThreshold), 0, 0);

  // If a region is selected, draw its bounding box overlay
  if (state.interactiveRegions && state.selectedRegionIndex !== null) {
    const selected = state.interactiveRegions[state.selectedRegionIndex];
    if (selected) {
      drawBoundingBox(selected.bbox);
    } else {
      state.selectedRegionIndex = null;
    }
  }

  canvas.hidden = false;
  el("empty").hidden = true;
  el("dimensions").textContent = `${canvas.width} × ${canvas.height}`;

  // Update legends using CSS classes (avoids CSP inline-style violations)
  const legendClassMap = {
    overlay: "legend-color-overlay",
    probability: "legend-color-prob",
    sensitivity: "legend-color-sensitivity",
    worst_case: "legend-color-worst",
    ela: "legend-color-ela",
    original: "legend-color-original"
  };
  const legendColor = el("legend-color");
  legendColor.className = legendClassMap[state.view] || "legend-color-overlay";

  const legends = {
    overlay: `Coral marks pixels above display threshold ${displayThreshold.toFixed(2)}`,
    probability: "White → green: model response from 0 to 1",
    sensitivity: `White → coral: JPEG-70 absolute response change (0 to 0.25+)`,
    worst_case: `White → orange: worst-case response shift across all 4 probe conditions`,
    ela: "White → purple: Error Level Analysis residual (amplified ×15)",
    original: "Original image · no overlay"
  };
  el("legend-text").textContent = state.result ? legends[state.view] : "Image loaded · ready for analysis";

  if (state.probability) {
    const stats = calculateMeasures(displayThreshold);
    state.interactiveStats = stats;
    el("area").textContent = pct(stats.flagged_pixel_fraction);
    el("area-sub").textContent = `at display threshold ${displayThreshold.toFixed(2)}`;

    if (stats.jpeg70_threshold_flip_fraction !== null) {
      el("flips").textContent = pct(stats.jpeg70_threshold_flip_fraction);
      el("flips-sub").textContent = `JPEG-70 flips at threshold ${displayThreshold.toFixed(2)}`;
    } else if (stats.worst_case_large_response_change_fraction !== null) {
      el("flips").textContent = pct(stats.worst_case_large_response_change_fraction);
      el("flips-sub").textContent = `absolute shift ≥ 0.10`;
    } else {
      el("flips").textContent = "Not checked";
      el("flips-sub").textContent = "compression probe disabled";
    }

    // Recompute interactive regions at the current display threshold
    const extraction = extractRegions(state.probability.data, state.original.width, state.original.height, displayThreshold, 10);
    state.interactiveRegions = extraction.regions;
    renderRegionsUI(extraction);
  }
}

async function selectFile(file) {
  if (!file || state.busy) return;
  try {
    if (file.size > 24 * 1024 * 1024) throw new Error("Choose an image smaller than 24 MB.");
    if (!["image/png", "image/jpeg", "image/webp"].includes(file.type)) {
      throw new Error("Choose a PNG, JPEG, or WebP image.");
    }
    const url = URL.createObjectURL(file);
    let pixels;
    try {
      pixels = await loadPixels(url);
    } finally {
      URL.revokeObjectURL(url);
    }

    state.file = file;
    state.original = pixels;
    state.result = null;
    state.probability = null;
    state.alternative = null;
    state.worst_case = null;
    state.ela_residual = null;
    state.selectedRegionIndex = null;
    state.interactiveRegions = [];
    state.interactiveStats = null;

    enableResults(false);
    chooseView("overlay");
    setBusy(false);

    el("file-info").textContent = file.name;
    ["area", "flips", "tiles"].forEach(id => el(id).textContent = "—");
    el("area-sub").textContent = "at display threshold";
    el("flips-sub").textContent = "compression response shifts";
    el("elapsed").textContent = "Ready for analysis";
    el("regions-card").hidden = true;

    message(
      file.name === "synthetic-sample.png"
        ? "Procedural unedited synthetic document. Highlights are false positives; use to explore thresholding and sensitivity."
        : "Image loaded. Run analysis to inspect model responses."
    );
  } catch (error) {
    message(error.message, true);
  }
}

el("file").addEventListener("change", event => selectFile(event.target.files[0]));

el("sample").addEventListener("click", async () => {
  try {
    const response = await fetch("/api/sample");
    if (!response.ok) throw new Error("Sample could not be loaded.");
    await selectFile(new File([await response.blob()], "synthetic-sample.png", { type: "image/png" }));
  } catch (error) {
    message(error.message, true);
  }
});

["dragover", "dragleave", "drop"].forEach(name => {
  el("drop").addEventListener(name, event => {
    event.preventDefault();
    el("drop").classList.toggle("drag", name === "dragover");
    if (name === "drop") selectFile(event.dataTransfer.files[0]);
  });
});

el("analyze").addEventListener("click", async () => {
  if (!state.file || state.busy) return;
  setBusy(true);
  enableResults(false);
  message("Analyzing locally at benchmark threshold 0.50. Large images may take longer.");

  try {
    const response = await fetch(`/api/analyze?probe=${el("probe").checked ? 1 : 0}&threshold=0.5`, {
      method: "POST",
      headers: { "Content-Type": state.file.type, "X-DocTrace-Request": "1" },
      body: state.file
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Analysis failed.");

    const url = content => `data:image/png;base64,${content}`;
    const [original, probability, alternative, worst_case, elaPixels] = await Promise.all([
      loadPixels(url(data.image)),
      loadPixels(url(data.probability)),
      data.alternative ? loadPixels(url(data.alternative)) : Promise.resolve(null),
      data.worst_case ? loadPixels(url(data.worst_case)) : Promise.resolve(null),
      data.ela ? loadPixels(url(data.ela)) : Promise.resolve(null),
    ]);

    state.original = original;
    state.probability = probability;
    state.alternative = alternative;
    state.worst_case = worst_case;
    state.ela_residual = elaPixels;
    state.result = data.report;
    state.selectedRegionIndex = null;

    // Reset threshold slider to 0.50
    el("threshold").value = 50;

    el("tiles").textContent = data.report.tiles;
    el("elapsed").textContent = `${data.report.elapsed_seconds.toFixed(2)} s · ${data.report.device.toUpperCase()}`;

    chooseView("overlay");
    enableResults(true);
    render();

    message(
      state.file.name === "synthetic-sample.png"
        ? "Analysis complete at analysis threshold 0.50. This is an unedited synthetic sample: highlighted areas are false positives."
        : "Analysis complete at analysis threshold 0.50. Inspect overlay, sensitivity, and ELA views before interpreting highlighted regions."
    );
  } catch (error) {
    message(error.message, true);
    enableResults(Boolean(state.result));
  } finally {
    setBusy(false);
  }
});

// Tab keyboard navigation (WAI-ARIA tabs pattern)
tabs.forEach((tab, index) => {
  tab.addEventListener("click", () => chooseView(tab.dataset.view));
  tab.addEventListener("keydown", e => {
    let newIndex = index;
    if (e.key === "ArrowRight") {
      newIndex = (index + 1) % tabs.length;
    } else if (e.key === "ArrowLeft") {
      newIndex = (index - 1 + tabs.length) % tabs.length;
    } else if (e.key === "Home") {
      newIndex = 0;
    } else if (e.key === "End") {
      newIndex = tabs.length - 1;
    } else {
      return;
    }
    e.preventDefault();
    const targetTab = tabs[newIndex];
    if (!targetTab.disabled) {
      targetTab.focus();
      chooseView(targetTab.dataset.view);
    }
  });
});

["threshold", "opacity"].forEach(id => el(id).addEventListener("input", render));

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function baseFileName() {
  return (state.file ? state.file.name : "document").replace(/\.[^.]*$/, "").replace(/[^a-zA-Z0-9_-]/g, "_") || "document";
}

function exportCanvasImage(view, filenameSuffix) {
  if (!state.result) return;
  const currentThreshold = Number(el("threshold").value) / 100;
  const out = document.createElement("canvas");
  out.width = canvas.width;
  out.height = canvas.height;
  const ctx = out.getContext("2d");
  ctx.putImageData(draw(view, currentThreshold), 0, 0);
  out.toBlob(blob => {
    if (blob) downloadBlob(blob, `${baseFileName()}-${filenameSuffix}.png`);
  }, "image/png");
}

function exportImageData(imgData, filenameSuffix) {
  if (!imgData) return;
  const out = document.createElement("canvas");
  out.width = imgData.width;
  out.height = imgData.height;
  out.getContext("2d").putImageData(imgData, 0, 0);
  out.toBlob(blob => {
    if (blob) downloadBlob(blob, `${baseFileName()}-${filenameSuffix}.png`);
  }, "image/png");
}

el("save-overlay").addEventListener("click", () => {
  const thresh = (Number(el("threshold").value) / 100).toFixed(2);
  exportCanvasImage("overlay", `overlay-thresh${thresh}`);
});

el("save-mask").addEventListener("click", () => {
  const thresh = (Number(el("threshold").value) / 100).toFixed(2);
  exportCanvasImage("mask", `mask-thresh${thresh}`);
});

el("save-ela-residual").addEventListener("click", () => {
  if (state.ela_residual) {
    exportImageData(state.ela_residual, "ela-residual");
  }
});

el("save-ela-rendered").addEventListener("click", () => {
  if (state.ela_residual) {
    exportCanvasImage("ela", "ela-rendered");
  }
});

el("save-sensitivity-worst").addEventListener("click", () => {
  if (state.worst_case) {
    exportImageData(state.worst_case, "sensitivity-worst-case");
  }
});

el("save-sensitivity-jpeg70").addEventListener("click", () => {
  if (state.alternative) {
    exportImageData(state.alternative, "sensitivity-jpeg70");
  }
});

el("save-report").addEventListener("click", () => {
  if (!state.result) return;
  const displayThreshold = Number(el("threshold").value) / 100;
  const currentStats = calculateMeasures(displayThreshold);
  const currentRegions = state.interactiveRegions || [];

  // Construct structured export with explicit server_analysis and interactive_inspection sections
  const reportExport = {
    schema_version: 3,
    model: state.result.model,
    epoch: state.result.epoch,
    checkpoint_sha256: state.result.checkpoint_sha256,
    device: state.result.device,
    tile_size: state.result.tile_size,
    stride: state.result.stride,
    width: state.result.width,
    height: state.result.height,
    tiles: state.result.tiles,
    elapsed_seconds: state.result.elapsed_seconds,
    input_name: state.file ? state.file.name : "unknown",

    // Immutable full-precision server analysis at analysis_threshold (0.50)
    server_analysis: {
      threshold: state.result.threshold,
      precision: "float32",
      flagged_pixel_fraction: state.result.flagged_pixel_fraction,
      regions: state.result.regions,
      total_regions_count: state.result.total_regions_count,
      returned_regions_count: state.result.returned_regions_count,
      regions_truncated: state.result.regions_truncated,
      ela: state.result.ela,
      probe: state.result.probe,
      multi_probe: state.result.multi_probe
    },

    // Interactive inspection recomputed on 8-bit quantized probability map at display_threshold
    interactive_inspection: {
      threshold: displayThreshold,
      quantization: "8-bit display map (values rounded to 1/255)",
      flagged_pixel_fraction: currentStats.flagged_pixel_fraction,
      regions: currentRegions,
      total_regions_count: currentRegions.length,
      returned_regions_count: currentRegions.length,
      regions_truncated: false,
      jpeg70_probe: currentStats.jpeg70_threshold_flip_fraction !== null ? {
        condition: "jpeg70",
        threshold: displayThreshold,
        threshold_flip_fraction: currentStats.jpeg70_threshold_flip_fraction
      } : null,
      large_response_change: currentStats.worst_case_large_response_change_fraction !== null ? {
        response_change_cutoff: currentStats.response_change_cutoff,
        worst_case_large_response_change_fraction: currentStats.worst_case_large_response_change_fraction
      } : null
    },

    interpretation: state.result.interpretation,
    limitations: state.result.limitations
  };

  downloadBlob(
    new Blob([JSON.stringify(reportExport, null, 2)], { type: "application/json" }),
    `${baseFileName()}-report.json`
  );
});

fetch("/api/status")
  .then(async response => {
    if (!response.ok) throw new Error("Model unavailable");
    const status = await response.json();
    state.ready = true;
    el("connection").textContent = `${status.device === "cuda" ? "GPU" : "CPU"} ready · local processing`;
    el("model-meta").textContent = `${status.model} · epoch ${status.epoch} · checkpoint ${status.checkpoint_sha256.slice(0, 12)}`;
    setBusy(false);
  })
  .catch(() => {
    el("connection").textContent = "Model offline";
    message("The model server is unavailable. Start the server using scripts/serve.py.", true);
  });
