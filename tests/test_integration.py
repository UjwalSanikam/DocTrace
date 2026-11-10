"""Integration tests exercising Predictor.analyze, sensitivity probe arithmetic,
threshold consistency, ELA metadata, and real checkpoint smoke verification.
"""
from io import BytesIO
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np
from PIL import Image
import torch

from doctrace.inference import Predictor, RESPONSE_CHANGE_CUTOFF
from doctrace.degradations import PROBE_CONDITIONS
from doctrace.synthetic import make_document


class DeterministicModel(torch.nn.Module):
    """Model that maps image input to a fixed predetermined response map."""
    def __init__(self, fixed_response: float = 0.6):
        super().__init__()
        self.fixed_response = fixed_response

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        # Returns logits such that sigmoid(logit) == fixed_response
        logit = np.log(self.fixed_response / (1.0 - self.fixed_response + 1e-7))
        b, _, h, w = tensor.shape
        return torch.full((b, 1, h, w), float(logit), dtype=tensor.dtype, device=tensor.device)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.img = Image.new("RGB", (64, 64), "white")

    def test_analyze_probe_off(self):
        predictor = Predictor.__new__(Predictor)
        predictor.device = torch.device("cpu")
        predictor.model = DeterministicModel(fixed_response=0.65)
        predictor.metadata = {
            "model": "deterministic_test",
            "epoch": 1,
            "checkpoint_sha256": "fakehash",
            "device": "cpu",
            "tile_size": 512,
            "stride": 384,
        }

        prob, alt, worst, ela, report = predictor.analyze(self.img, probe=False, threshold=0.5)

        self.assertEqual(prob.shape, (64, 64))
        self.assertIsNone(alt)
        self.assertIsNone(worst)
        self.assertEqual(ela.shape, (64, 64, 3))
        self.assertEqual(ela.dtype, np.uint8)

        self.assertEqual(report["schema_version"], 3)
        self.assertEqual(report["threshold"], 0.5)
        self.assertAlmostEqual(report["flagged_pixel_fraction"], 1.0)
        self.assertIsNone(report["probe"])
        self.assertIsNone(report["multi_probe"])

        # ELA parameters
        self.assertEqual(report["ela"]["quality"], 90)
        self.assertEqual(report["ela"]["subsampling"], "4:2:0")
        self.assertEqual(report["ela"]["amplification"], 15)
        self.assertEqual(report["ela"]["clipping"], [0, 255])
        self.assertIn("Error Level Analysis", report["ela"]["interpretation"])

    def test_analyze_probe_on(self):
        predictor = Predictor.__new__(Predictor)
        predictor.device = torch.device("cpu")
        predictor.model = DeterministicModel(fixed_response=0.65)
        predictor.metadata = {
            "model": "deterministic_test",
            "epoch": 1,
            "checkpoint_sha256": "fakehash",
            "device": "cpu",
            "tile_size": 512,
            "stride": 384,
        }

        prob, alt, worst, ela, report = predictor.analyze(self.img, probe=True, threshold=0.5)

        self.assertIsNotNone(alt)
        self.assertIsNotNone(worst)
        self.assertIsNotNone(report["probe"])
        self.assertIsNotNone(report["multi_probe"])

        # Since model produces constant 0.65 for clean and all degradations:
        self.assertAlmostEqual(report["multi_probe"]["worst_case_mean_change"], 0.0, places=5)
        self.assertEqual(report["multi_probe"]["worst_case_threshold_flip_fraction"], 0.0)
        self.assertEqual(report["multi_probe"]["worst_case_large_response_change_fraction"], 0.0)
        self.assertEqual(set(report["multi_probe"]["conditions_tested"]), set(PROBE_CONDITIONS))

    def test_threshold_flip_and_large_change_arithmetic(self):
        """Verify the exact regression examples:
        1. 0.60 -> 0.85 at threshold 0.50: large response change (0.25 >= 0.10), no threshold flip.
        2. 0.49 -> 0.51 at threshold 0.50: threshold flip, small response change (0.02 < 0.10).
        3. Identical maps: both metrics zero.
        4. Several conditions union calculation.
        """
        # Create a predictor that returns tailored responses based on image pixel values
        class ControlledModel(torch.nn.Module):
            def forward(self, tensor: torch.Tensor) -> torch.Tensor:
                # Use red channel directly as target probability
                p = torch.clamp(tensor[:, :1], 0.001, 0.999)
                return torch.log(p / (1.0 - p))

        # We test metric arithmetic directly on probability arrays
        threshold = 0.50
        p_clean = np.array([[0.60, 0.49, 0.70, 0.40]], dtype=np.float32)
        p_cond1 = np.array([[0.85, 0.51, 0.70, 0.40]], dtype=np.float32) # px0: change=0.25, no flip; px1: change=0.02, FLIP
        p_cond2 = np.array([[0.60, 0.49, 0.30, 0.40]], dtype=np.float32) # px2: change=0.40, FLIP; others: no change

        # Condition 1:
        abs1 = np.abs(p_clean - p_cond1)
        flip1 = (p_clean >= threshold) != (p_cond1 >= threshold)
        large1 = abs1 >= RESPONSE_CHANGE_CUTOFF

        # Pixel 0: 0.60 -> 0.85: large change, NO flip
        self.assertFalse(flip1[0, 0])
        self.assertTrue(large1[0, 0])

        # Pixel 1: 0.49 -> 0.51: threshold flip, NO large change
        self.assertTrue(flip1[0, 1])
        self.assertFalse(large1[0, 1])

        # Condition 2:
        abs2 = np.abs(p_clean - p_cond2)
        flip2 = (p_clean >= threshold) != (p_cond2 >= threshold)
        large2 = abs2 >= RESPONSE_CHANGE_CUTOFF

        # Union across conditions:
        worst_case_abs = np.maximum(abs1, abs2)
        union_flips = flip1 | flip2
        union_large = worst_case_abs >= RESPONSE_CHANGE_CUTOFF

        # Pixel 0: no flip under either condition
        # Pixel 1: flipped under cond1
        # Pixel 2: flipped under cond2
        # Pixel 3: no flip under either
        # -> Total flips in union = 2 / 4 = 0.50
        self.assertEqual(float(union_flips.mean()), 0.50)

        # Large change union:
        # Pixel 0: change 0.25 >= 0.10 (cond1)
        # Pixel 1: change 0.02 < 0.10
        # Pixel 2: change 0.40 >= 0.10 (cond2)
        # Pixel 3: change 0.00
        # -> Total large change = 2 / 4 = 0.50
        self.assertEqual(float(union_large.mean()), 0.50)

    def test_region_and_report_threshold_consistency(self):
        """Analyze at 0.50 and 0.90; check that every threshold-dependent field reflects the threshold."""
        predictor = Predictor.__new__(Predictor)
        predictor.device = torch.device("cpu")
        predictor.model = DeterministicModel(fixed_response=0.75)
        predictor.metadata = {
            "model": "deterministic_test",
            "epoch": 1,
            "checkpoint_sha256": "fakehash",
            "device": "cpu",
            "tile_size": 512,
            "stride": 384,
        }

        # At threshold 0.50 (prob=0.75 > 0.50): all pixels flagged, 1 region
        _, _, _, _, report50 = predictor.analyze(self.img, probe=False, threshold=0.50)
        self.assertEqual(report50["threshold"], 0.50)
        self.assertAlmostEqual(report50["flagged_pixel_fraction"], 1.0)
        self.assertEqual(len(report50["regions"]), 1)
        self.assertEqual(report50["total_regions_count"], 1)

        # At threshold 0.90 (prob=0.75 < 0.90): 0 pixels flagged, 0 regions
        _, _, _, _, report90 = predictor.analyze(self.img, probe=False, threshold=0.90)
        self.assertEqual(report90["threshold"], 0.90)
        self.assertEqual(report90["flagged_pixel_fraction"], 0.0)
        self.assertEqual(len(report90["regions"]), 0)
        self.assertEqual(report90["total_regions_count"], 0)

    def test_checkpoint_smoke_if_present(self):
        """Local smoke test with actual exported weights (outputs/release_v1/model.pt).
        Cleanly skips if weights are not present.
        """
        checkpoint_path = ROOT / "outputs" / "release_v1" / "model.pt"
        if not checkpoint_path.is_file():
            raise unittest.SkipTest("No local exported checkpoint found at outputs/release_v1/model.pt")

        predictor = Predictor(checkpoint_path, device="cpu")
        sample_img = make_document(42)
        prob, alt, worst, ela, report = predictor.analyze(sample_img, probe=True, threshold=0.5)

        self.assertEqual(prob.shape, (sample_img.height, sample_img.width))
        self.assertTrue(np.all(np.isfinite(prob)))
        self.assertTrue(np.all((prob >= 0.0) & (prob <= 1.0)))
        self.assertIsNotNone(alt)
        self.assertIsNotNone(worst)
        self.assertIsNotNone(ela)
        self.assertEqual(report["schema_version"], 3)
        self.assertEqual(report["threshold"], 0.5)
        self.assertIn("regions", report)
        self.assertIn("multi_probe", report)
        self.assertIn("ela", report)


if __name__ == "__main__":
    unittest.main()
