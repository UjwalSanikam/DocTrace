/**
 * Pure analysis and report-building functions for DocTrace.
 * Shared between the browser runtime and headless/Node regression tests.
 */
"use strict";

/**
 * Fast 4-connectivity connected-component extraction on quantized 8-bit probability map.
 *
 * @param {Uint8ClampedArray|Uint8Array} probData - RGBA or flat 8-bit array (prob in channel 0: idx * 4).
 * @param {number} width - image width in pixels.
 * @param {number} height - image height in pixels.
 * @param {number} threshold - binarization threshold in (0, 1).
 * @param {number|null} maxRegions - max regions to return (default 10). If null, returns all.
 * @returns {Object} { total_count, returned_count, truncated, threshold, regions }
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

  // Deterministic sort: area desc, mean response desc, yMin asc, xMin asc, yMax asc, xMax asc
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

/**
 * Compute flagged fraction, JPEG-70 threshold flips, and large response changes.
 */
function calculateMeasures(probData, altData, worstData, width, height, threshold) {
  const cutoff = Math.ceil(threshold * 255);
  const count = width * height;
  let positive = 0;
  let jpeg70Flips = 0;
  let largeResponseChanges = 0;
  const changeCutoff = Math.ceil(0.10 * 255);

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

/**
 * Build the structured export report clearly separating immutable server analysis
 * from interactive inspection at display threshold.
 */
function buildReportExport({ serverAnalysis, interactiveStats, interactiveRegionResult, displayThreshold, inputName = "unknown" }) {
  if (!serverAnalysis) return null;

  const regionsList = interactiveRegionResult && Array.isArray(interactiveRegionResult.regions)
    ? interactiveRegionResult.regions
    : [];
  const totalCount = interactiveRegionResult && typeof interactiveRegionResult.total_count === "number"
    ? interactiveRegionResult.total_count
    : regionsList.length;
  const returnedCount = interactiveRegionResult && typeof interactiveRegionResult.returned_count === "number"
    ? interactiveRegionResult.returned_count
    : regionsList.length;
  const truncated = interactiveRegionResult && typeof interactiveRegionResult.truncated === "boolean"
    ? interactiveRegionResult.truncated
    : totalCount > returnedCount;

  return {
    schema_version: 3,
    model: serverAnalysis.model,
    epoch: serverAnalysis.epoch,
    checkpoint_sha256: serverAnalysis.checkpoint_sha256,
    device: serverAnalysis.device,
    tile_size: serverAnalysis.tile_size,
    stride: serverAnalysis.stride,
    width: serverAnalysis.width,
    height: serverAnalysis.height,
    tiles: serverAnalysis.tiles,
    elapsed_seconds: serverAnalysis.elapsed_seconds,
    input_name: inputName,

    // Immutable full-precision server analysis at analysis_threshold (0.50)
    server_analysis: {
      threshold: serverAnalysis.threshold,
      precision: "float32",
      flagged_pixel_fraction: serverAnalysis.flagged_pixel_fraction,
      regions: serverAnalysis.regions,
      total_regions_count: serverAnalysis.total_regions_count,
      returned_regions_count: serverAnalysis.returned_regions_count,
      regions_truncated: serverAnalysis.regions_truncated,
      ela: serverAnalysis.ela,
      probe: serverAnalysis.probe,
      multi_probe: serverAnalysis.multi_probe
    },

    // Interactive inspection recomputed on 8-bit quantized probability map at display_threshold
    interactive_inspection: {
      threshold: displayThreshold,
      quantization: "8-bit display map (values rounded to 1/255)",
      flagged_pixel_fraction: interactiveStats ? interactiveStats.flagged_pixel_fraction : null,
      regions: regionsList,
      total_regions_count: totalCount,
      returned_regions_count: returnedCount,
      regions_truncated: truncated,
      jpeg70_probe: (interactiveStats && interactiveStats.jpeg70_threshold_flip_fraction !== null) ? {
        condition: "jpeg70",
        threshold: displayThreshold,
        threshold_flip_fraction: interactiveStats.jpeg70_threshold_flip_fraction
      } : null,
      large_response_change: (interactiveStats && interactiveStats.worst_case_large_response_change_fraction !== null) ? {
        response_change_cutoff: interactiveStats.response_change_cutoff,
        worst_case_large_response_change_fraction: interactiveStats.worst_case_large_response_change_fraction
      } : null
    },

    interpretation: serverAnalysis.interpretation,
    limitations: serverAnalysis.limitations
  };
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    extractRegions,
    calculateMeasures,
    buildReportExport
  };
}
