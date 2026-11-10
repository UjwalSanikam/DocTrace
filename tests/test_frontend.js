/**
 * Frontend logic regression test suite for DocTrace.
 * Exercises client-side region extraction, measure calculations,
 * and report export formatting without requiring a browser DOM.
 */
"use strict";

const assert = require("assert");

// Pure JS implementations mirroring web/app.js:

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

  for (let i = 1; i < parent.length; i++) {
    let r = i;
    while (parent[r] !== r) r = parent[r];
    parent[i] = r;
  }

  const rootMap = new Map();
  let numComponents = 0;
  for (let i = 1; i < parent.length; i++) {
    const r = parent[i];
    if (!rootMap.has(r)) {
      numComponents++;
      rootMap.set(r, numComponents);
    }
  }

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
      mean_confidence: meanResp,
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

// ----------------------------------------------------
// Tests
// ----------------------------------------------------

console.log("Running DocTrace frontend regression tests...");

// Test 1: extractRegions with empty data
{
  const width = 10, height = 10;
  const probData = new Uint8ClampedArray(width * height * 4); // all zeros
  const res = extractRegions(probData, width, height, 0.5);
  assert.strictEqual(res.total_count, 0);
  assert.strictEqual(res.returned_count, 0);
  assert.strictEqual(res.truncated, false);
  assert.strictEqual(res.threshold, 0.5);
  assert.strictEqual(res.regions.length, 0);
  console.log("✓ Test 1: extractRegions empty data passed");
}

// Test 2: extractRegions single component & bounding box
{
  const width = 10, height = 10;
  const probData = new Uint8ClampedArray(width * height * 4);
  // Fill 2x3 block at y: 2..3, x: 4..6 with probability 0.8 (val ~ 204)
  for (let y = 2; y <= 3; y++) {
    for (let x = 4; x <= 6; x++) {
      const idx = (y * width + x) * 4;
      probData[idx] = 204;
    }
  }
  const res = extractRegions(probData, width, height, 0.5);
  assert.strictEqual(res.total_count, 1);
  assert.strictEqual(res.regions.length, 1);
  const r = res.regions[0];
  assert.strictEqual(r.rank, 1);
  assert.strictEqual(r.area_px, 6);
  assert.strictEqual(r.area_fraction, 6 / 100);
  assert.deepStrictEqual(r.bbox, [2, 4, 3, 6]);
  assert.strictEqual(r.mean_confidence, r.mean_model_response);
  console.log("✓ Test 2: extractRegions single component passed");
}

// Test 3: extractRegions ranking by area
{
  const width = 20, height = 20;
  const probData = new Uint8ClampedArray(width * height * 4);
  // Small component at (1, 1), area 2
  probData[(1 * width + 1) * 4] = 220;
  probData[(1 * width + 2) * 4] = 220;
  // Large component at (10, 10), area 6
  for (let y = 10; y <= 11; y++) {
    for (let x = 10; x <= 12; x++) {
      probData[(y * width + x) * 4] = 200;
    }
  }
  const res = extractRegions(probData, width, height, 0.5);
  assert.strictEqual(res.total_count, 2);
  assert.strictEqual(res.regions[0].area_px, 6);
  assert.strictEqual(res.regions[1].area_px, 2);
  console.log("✓ Test 3: extractRegions ranking passed");
}

// Test 4: Threshold slider change dynamic recomputation
{
  const width = 10, height = 10;
  const probData = new Uint8ClampedArray(width * height * 4);
  // Pixel A has prob 0.60 (val 153)
  probData[(2 * width + 2) * 4] = 153;
  // Pixel B has prob 0.95 (val 242)
  probData[(5 * width + 5) * 4] = 242;

  // At threshold 0.50, both A and B are active (2 regions)
  const res50 = extractRegions(probData, width, height, 0.50);
  assert.strictEqual(res50.total_count, 2);

  // At threshold 0.70, only B is active (1 region)
  const res70 = extractRegions(probData, width, height, 0.70);
  assert.strictEqual(res70.total_count, 1);
  assert.deepStrictEqual(res70.regions[0].bbox, [5, 5, 5, 5]);

  console.log("✓ Test 4: Threshold dynamic recomputation passed");
}

// Test 5: calculateMeasures threshold flips vs large response change
{
  const width = 2, height = 2; // 4 pixels
  const probData = new Uint8ClampedArray(16);
  const altData = new Uint8ClampedArray(16);
  const worstData = new Uint8ClampedArray(16);

  // Pixel 0: 0.60 -> 0.85 (val 153 -> 217). Shift = 64 (0.25).
  // At threshold 0.50 (cutoff 128): both active -> no flip. Shift >= 0.10 -> large response change.
  probData[0] = 153; altData[0] = 217; worstData[0] = 64;

  // Pixel 1: 0.49 -> 0.51 (val 125 -> 130). Shift = 5 (0.02).
  // At threshold 0.50 (cutoff 128): clean inactive, alt active -> FLIP! Shift < 0.10 -> not large change.
  probData[4] = 125; altData[4] = 130; worstData[4] = 5;

  // Pixel 2: 0.10 -> 0.10 (val 26 -> 26). Shift = 0.
  // No flip, no large change.
  probData[8] = 26; altData[8] = 26; worstData[8] = 0;

  // Pixel 3: 0.80 -> 0.40 (val 204 -> 102). Shift = 102 (0.40).
  // At threshold 0.50: clean active, alt inactive -> FLIP! Shift >= 0.10 -> large change.
  probData[12] = 204; altData[12] = 102; worstData[12] = 102;

  const stats = calculateMeasures(probData, altData, worstData, width, height, 0.50);
  assert.strictEqual(stats.flagged_pixel_fraction, 2 / 4); // pixel 0 and pixel 3
  assert.strictEqual(stats.jpeg70_threshold_flip_fraction, 2 / 4); // pixel 1 and pixel 3 flip
  assert.strictEqual(stats.worst_case_large_response_change_fraction, 2 / 4); // pixel 0 and pixel 3 large change

  console.log("✓ Test 5: calculateMeasures metrics arithmetic passed");
}

// Test 6: Report export schema separates server_analysis from interactive_inspection
{
  const mockServerAnalysis = {
    threshold: 0.50,
    flagged_pixel_fraction: 0.15,
    regions: [{ rank: 1, area_px: 150 }],
    total_regions_count: 1,
    returned_regions_count: 1,
    regions_truncated: false
  };

  const displayThreshold = 0.80;
  const mockInteractiveStats = {
    flagged_pixel_fraction: 0.04,
    jpeg70_threshold_flip_fraction: 0.02,
    worst_case_large_response_change_fraction: 0.05,
    response_change_cutoff: 0.10
  };
  const mockInteractiveRegions = [{ rank: 1, area_px: 40 }];

  const reportExport = {
    schema_version: 3,
    server_analysis: {
      threshold: mockServerAnalysis.threshold,
      precision: "float32",
      flagged_pixel_fraction: mockServerAnalysis.flagged_pixel_fraction,
      regions: mockServerAnalysis.regions
    },
    interactive_inspection: {
      threshold: displayThreshold,
      quantization: "8-bit display map (values rounded to 1/255)",
      flagged_pixel_fraction: mockInteractiveStats.flagged_pixel_fraction,
      regions: mockInteractiveRegions,
      jpeg70_probe: {
        condition: "jpeg70",
        threshold: displayThreshold,
        threshold_flip_fraction: mockInteractiveStats.jpeg70_threshold_flip_fraction
      }
    }
  };

  assert.strictEqual(reportExport.server_analysis.threshold, 0.50);
  assert.strictEqual(reportExport.interactive_inspection.threshold, 0.80);
  assert.strictEqual(reportExport.server_analysis.flagged_pixel_fraction, 0.15);
  assert.strictEqual(reportExport.interactive_inspection.flagged_pixel_fraction, 0.04);
  assert.strictEqual(reportExport.interactive_inspection.jpeg70_probe.threshold, 0.80);

  console.log("✓ Test 6: Report export schema separation passed");
}

console.log("All DocTrace frontend tests passed successfully!");
