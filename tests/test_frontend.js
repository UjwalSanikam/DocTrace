/**
 * Frontend logic regression test suite for DocTrace.
 * Imports actual production logic from web/analysis.js.
 */
"use strict";

const assert = require("assert");
const { extractRegions, calculateMeasures, buildReportExport } = require("../web/analysis.js");

console.log("Running DocTrace frontend regression tests (using production web/analysis.js)...");

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

// Test 6: Regression with >= 12 disconnected regions and limit of 10
{
  const width = 30, height = 30;
  const probData = new Uint8ClampedArray(width * height * 4);
  // Create exactly 12 isolated single pixels, well separated (step of 2 along x and y)
  // Ensure each pixel has prob 0.90 (val 230)
  for (let i = 0; i < 12; i++) {
    const y = (i % 4) * 6 + 1;
    const x = Math.floor(i / 4) * 8 + 1;
    const idx = (y * width + x) * 4;
    probData[idx] = 230;
  }

  const res = extractRegions(probData, width, height, 0.50, 10);
  assert.strictEqual(res.total_count, 12, "Should find 12 total components");
  assert.strictEqual(res.returned_count, 10, "Should cap returned list to 10");
  assert.strictEqual(res.truncated, true, "Should report truncated: true");
  assert.strictEqual(res.regions.length, 10, "regions array should have length 10");

  const mockServerAnalysis = {
    schema_version: 3,
    model: "ResNet18UNet",
    epoch: 10,
    checkpoint_sha256: "testsha256",
    device: "cpu",
    tile_size: 512,
    stride: 384,
    width,
    height,
    tiles: 1,
    elapsed_seconds: 0.12,
    threshold: 0.50,
    flagged_pixel_fraction: 12 / (width * height),
    regions: [{ rank: 1, area_px: 1, area_fraction: 1 / 900, mean_model_response: 0.9, mean_confidence: 0.9, bbox: [1, 1, 1, 1] }],
    total_regions_count: 12,
    returned_regions_count: 1,
    regions_truncated: true,
    ela: { quality: 90 },
    probe: null,
    multi_probe: null,
    interpretation: "prototype",
    limitations: "limitations"
  };

  const mockStats = {
    threshold: 0.50,
    flagged_pixel_fraction: 12 / (width * height),
    jpeg70_threshold_flip_fraction: null,
    worst_case_large_response_change_fraction: null,
    response_change_cutoff: 0.10
  };

  const reportExport = buildReportExport({
    serverAnalysis: mockServerAnalysis,
    interactiveStats: mockStats,
    interactiveRegionResult: res,
    displayThreshold: 0.50,
    inputName: "doc12.png"
  });

  // Verify full client-side extraction result is preserved in export:
  assert.strictEqual(reportExport.interactive_inspection.total_regions_count, 12, "Exported total_regions_count must be 12 (not derived from capped array length 10)");
  assert.strictEqual(reportExport.interactive_inspection.returned_regions_count, 10, "Exported returned_regions_count must be 10");
  assert.strictEqual(reportExport.interactive_inspection.regions_truncated, true, "Exported regions_truncated must be true");
  assert.strictEqual(reportExport.interactive_inspection.regions.length, 10);
  assert.strictEqual(reportExport.input_name, "doc12.png");

  console.log("✓ Test 6: 12 regions with limit 10 regression passed (total=12, returned=10, truncated=true)");
}

// Test 7: Real report builder preserves immutable server-analysis fields
{
  const mockServerAnalysis = {
    schema_version: 3,
    model: "ResNet18UNet",
    epoch: 8,
    checkpoint_sha256: "abc123hash",
    device: "cuda",
    tile_size: 512,
    stride: 384,
    width: 512,
    height: 512,
    tiles: 1,
    elapsed_seconds: 0.35,
    threshold: 0.50,
    flagged_pixel_fraction: 0.05,
    regions: [{ rank: 1, area_px: 1300 }],
    total_regions_count: 1,
    returned_regions_count: 1,
    regions_truncated: false,
    ela: { quality: 90, amplification: 15 },
    probe: { condition: "jpeg70", threshold: 0.50 },
    multi_probe: { threshold: 0.50, worst_case_threshold_flip_fraction: 0.03 },
    interpretation: "test interpretation",
    limitations: "test limitations"
  };

  const displayThreshold = 0.85;
  const mockInteractiveStats = {
    threshold: displayThreshold,
    flagged_pixel_fraction: 0.008,
    jpeg70_threshold_flip_fraction: 0.005,
    worst_case_large_response_change_fraction: 0.02,
    response_change_cutoff: 0.10
  };
  const mockInteractiveExtraction = {
    total_count: 3,
    returned_count: 3,
    truncated: false,
    threshold: displayThreshold,
    regions: [{ rank: 1, area_px: 200 }, { rank: 2, area_px: 100 }, { rank: 3, area_px: 50 }]
  };

  const reportExport = buildReportExport({
    serverAnalysis: mockServerAnalysis,
    interactiveStats: mockInteractiveStats,
    interactiveRegionResult: mockInteractiveExtraction,
    displayThreshold: displayThreshold,
    inputName: "receipt.png"
  });

  // Verify server_analysis remains strictly immutable at 0.50
  assert.strictEqual(reportExport.server_analysis.threshold, 0.50);
  assert.strictEqual(reportExport.server_analysis.precision, "float32");
  assert.strictEqual(reportExport.server_analysis.flagged_pixel_fraction, 0.05);
  assert.strictEqual(reportExport.server_analysis.total_regions_count, 1);
  assert.deepStrictEqual(reportExport.server_analysis.regions, mockServerAnalysis.regions);
  assert.deepStrictEqual(reportExport.server_analysis.ela, mockServerAnalysis.ela);
  assert.deepStrictEqual(reportExport.server_analysis.probe, mockServerAnalysis.probe);
  assert.deepStrictEqual(reportExport.server_analysis.multi_probe, mockServerAnalysis.multi_probe);

  // Verify interactive_inspection corresponds to displayThreshold 0.85
  assert.strictEqual(reportExport.interactive_inspection.threshold, 0.85);
  assert.strictEqual(reportExport.interactive_inspection.flagged_pixel_fraction, 0.008);
  assert.strictEqual(reportExport.interactive_inspection.total_regions_count, 3);
  assert.strictEqual(reportExport.interactive_inspection.returned_regions_count, 3);
  assert.strictEqual(reportExport.interactive_inspection.regions_truncated, false);
  assert.strictEqual(reportExport.interactive_inspection.jpeg70_probe.threshold, 0.85);
  assert.strictEqual(reportExport.interactive_inspection.jpeg70_probe.threshold_flip_fraction, 0.005);

  console.log("✓ Test 7: Immutable server-analysis fields preserved across display threshold changes passed");
}

console.log("All DocTrace frontend tests passed successfully!");
