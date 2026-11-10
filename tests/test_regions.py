"""Unit and benchmark tests for connected-component region extraction."""
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np
from doctrace.regions import summarize_regions, RegionList


class RegionExtractionTests(unittest.TestCase):
    def test_empty_mask(self):
        prob = np.zeros((30, 40), dtype=np.float32)
        res = summarize_regions(prob, threshold=0.5)
        self.assertIsInstance(res, RegionList)
        self.assertEqual(len(res), 0)
        self.assertEqual(res.total_count, 0)
        self.assertEqual(res.returned_count, 0)
        self.assertFalse(res.truncated)
        self.assertEqual(res.threshold, 0.5)

    def test_full_mask(self):
        h, w = 25, 35
        prob = np.full((h, w), 0.75, dtype=np.float32)
        res = summarize_regions(prob, threshold=0.5)
        self.assertEqual(len(res), 1)
        self.assertEqual(res.total_count, 1)
        self.assertEqual(res.returned_count, 1)
        self.assertFalse(res.truncated)
        r0 = res[0]
        self.assertEqual(r0["rank"], 1)
        self.assertEqual(r0["area_px"], h * w)
        self.assertAlmostEqual(r0["area_fraction"], 1.0)
        self.assertAlmostEqual(r0["mean_model_response"], 0.75, places=4)
        self.assertAlmostEqual(r0["mean_confidence"], 0.75, places=4)
        self.assertEqual(r0["bbox"], [0, 0, h - 1, w - 1])

    def test_diagonal_pixels_are_not_connected(self):
        # 4-connectivity rule: diagonally adjacent pixels must form distinct components.
        prob = np.zeros((4, 4), dtype=np.float32)
        prob[0, 0] = 0.9
        prob[1, 1] = 0.9
        prob[2, 2] = 0.9
        prob[3, 3] = 0.9
        res = summarize_regions(prob, threshold=0.5, max_regions=10)
        self.assertEqual(res.total_count, 4)
        self.assertEqual(len(res), 4)
        for r in res:
            self.assertEqual(r["area_px"], 1)

    def test_border_components(self):
        prob = np.zeros((10, 10), dtype=np.float32)
        # Top-left corner
        prob[0, 0] = 0.9
        # Bottom-right corner
        prob[9, 9] = 0.8
        # Center 2x2
        prob[4:6, 4:6] = 0.95
        res = summarize_regions(prob, threshold=0.5)
        self.assertEqual(res.total_count, 3)
        # Largest is center (4 px)
        self.assertEqual(res[0]["area_px"], 4)
        self.assertEqual(res[0]["bbox"], [4, 4, 5, 5])
        # Check corner bboxes
        bboxes = [r["bbox"] for r in res]
        self.assertIn([0, 0, 0, 0], bboxes)
        self.assertIn([9, 9, 9, 9], bboxes)

    def test_merged_equivalences(self):
        # U-shaped component: starts as two branches at top, connects at bottom.
        # [1, 0, 1]
        # [1, 0, 1]
        # [1, 1, 1]
        prob = np.zeros((5, 5), dtype=np.float32)
        prob[0, 0] = 0.8; prob[0, 2] = 0.8
        prob[1, 0] = 0.8; prob[1, 2] = 0.8
        prob[2, 0] = 0.8; prob[2, 1] = 0.8; prob[2, 2] = 0.8
        res = summarize_regions(prob, threshold=0.5)
        self.assertEqual(res.total_count, 1)
        self.assertEqual(res[0]["area_px"], 7)
        self.assertEqual(res[0]["bbox"], [0, 0, 2, 2])

    def test_equal_size_ties_deterministic(self):
        prob = np.zeros((10, 10), dtype=np.float32)
        # Component A at (1, 1), area 2, mean 0.9
        prob[1, 1:3] = 0.9
        # Component B at (5, 5), area 2, mean 0.7
        prob[5, 5:7] = 0.7
        # Component C at (1, 5), area 2, mean 0.9 (same area and mean as A, but x_min=5 > x_min=1)
        prob[1, 5:7] = 0.9

        res = summarize_regions(prob, threshold=0.5)
        self.assertEqual(res.total_count, 3)
        # Rank 1: A (mean 0.9, x_min 1)
        self.assertEqual(res[0]["bbox"], [1, 1, 1, 2])
        # Rank 2: C (mean 0.9, x_min 5)
        self.assertEqual(res[1]["bbox"], [1, 5, 1, 6])
        # Rank 3: B (mean 0.7)
        self.assertEqual(res[2]["bbox"], [5, 5, 5, 6])

    def test_fragmented_mask_and_max_regions(self):
        # 20 isolated components
        prob = np.zeros((40, 40), dtype=np.float32)
        for i in range(20):
            prob[i * 2, 0] = 0.6 + (i * 0.01)
        res = summarize_regions(prob, threshold=0.5, max_regions=5)
        self.assertEqual(res.total_count, 20)
        self.assertEqual(res.returned_count, 5)
        self.assertEqual(len(res), 5)
        self.assertTrue(res.truncated)
        # Verify unconstrained returns all
        res_all = summarize_regions(prob, threshold=0.5, max_regions=None)
        self.assertEqual(res_all.total_count, 20)
        self.assertEqual(res_all.returned_count, 20)
        self.assertFalse(res_all.truncated)

    def test_invalid_parameters(self):
        prob = np.zeros((10, 10), dtype=np.float32)
        with self.assertRaises(TypeError):
            summarize_regions("not an array", threshold=0.5)
        with self.assertRaises(ValueError):
            summarize_regions(np.zeros((10, 10, 2), dtype=np.float32), threshold=0.5)
        with self.assertRaises(ValueError):
            summarize_regions(np.zeros((0, 10), dtype=np.float32), threshold=0.5)

        # Non-finite values
        prob_nan = prob.copy()
        prob_nan[2, 2] = np.nan
        with self.assertRaises(ValueError):
            summarize_regions(prob_nan, threshold=0.5)

        prob_inf = prob.copy()
        prob_inf[2, 2] = np.inf
        with self.assertRaises(ValueError):
            summarize_regions(prob_inf, threshold=0.5)

        # Invalid thresholds
        with self.assertRaises(ValueError):
            summarize_regions(prob, threshold=0.0)
        with self.assertRaises(ValueError):
            summarize_regions(prob, threshold=1.0)
        with self.assertRaises(ValueError):
            summarize_regions(prob, threshold=-0.1)
        with self.assertRaises(TypeError):
            summarize_regions(prob, threshold="0.5")

        # Invalid max_regions
        with self.assertRaises(ValueError):
            summarize_regions(prob, threshold=0.5, max_regions=-1)
        with self.assertRaises(TypeError):
            summarize_regions(prob, threshold=0.5, max_regions="10")


class RegionBenchmarkTests(unittest.TestCase):
    def test_benchmark_512x512_fragmented(self):
        rng = np.random.default_rng(42)
        # 512x512 with random noise around 0.5 threshold creates thousands of micro-components
        prob = rng.uniform(0.0, 1.0, (512, 512)).astype(np.float32)
        t0 = time.perf_counter()
        res = summarize_regions(prob, threshold=0.5, max_regions=10)
        elapsed = time.perf_counter() - t0
        self.assertGreater(res.total_count, 500)
        self.assertEqual(len(res), 10)
        self.assertTrue(res.truncated)
        # Avoid brittle timing assertions in CI; assert generous upper bound (< 10.0s)
        self.assertLess(elapsed, 10.0, f"512x512 extraction took too long: {elapsed:.2f}s")

    def test_benchmark_multimegapixel(self):
        # 2048 x 1536 (~3.14 MP), simulating full-page scan with scattered annotations
        prob = np.zeros((2048, 1536), dtype=np.float32)
        # Stamp several hundred tampering patches
        for y in range(50, 1900, 80):
            for x in range(50, 1400, 70):
                prob[y:y + 15, x:x + 25] = 0.85
        t0 = time.perf_counter()
        res = summarize_regions(prob, threshold=0.5, max_regions=10)
        elapsed = time.perf_counter() - t0
        self.assertGreater(res.total_count, 100)
        self.assertEqual(len(res), 10)
        self.assertLess(elapsed, 10.0, f"Multi-megapixel extraction took too long: {elapsed:.2f}s")


if __name__ == "__main__":
    unittest.main()
