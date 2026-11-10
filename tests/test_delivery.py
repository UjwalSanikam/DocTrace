"""CPU tests for decoding, geometry, augmentations, ELA, regions, and local HTTP boundaries."""
from io import BytesIO
import json
from pathlib import Path
import sys
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import numpy as np
from PIL import Image
import torch
from doctrace.inference import Predictor, decode_image, starts, png_bytes
from doctrace.degradations import degrade, ela, CONDITIONS, PROBE_CONDITIONS
from doctrace.augmentation import JpegTrainingDataset
from doctrace.synthetic import make_document
from doctrace.regions import summarize_regions
from serve import make_handler


class PixelModel(torch.nn.Module):
    def forward(self, tensor):
        return tensor[:, :1] * 2 - 1


class DeliveryTests(unittest.TestCase):
    def test_tile_overlap_reconstructs_coordinates(self):
        predictor = Predictor.__new__(Predictor)
        predictor.device, predictor.model = torch.device("cpu"), PixelModel()
        rng = np.random.default_rng(10)
        for shape in ((19, 27, 3), (617, 769, 3), (512, 512, 3)):
            rgb = rng.integers(0, 256, shape, dtype=np.uint8)
            actual = predictor.predict(Image.fromarray(rgb))
            expected = 1 / (1 + np.exp(-(rgb[:, :, 0].astype(np.float32)/255*2-1)))
            np.testing.assert_allclose(actual, expected, atol=2e-7)
            self.assertEqual(actual.shape, shape[:2])

    def test_tiling_boundaries(self):
        for length in (1, 511, 512, 513, 769, 1001, 8192):
            covered = np.zeros(max(length, 512), dtype=bool)
            for x in starts(length):
                covered[x:x+512] = True
            self.assertTrue(covered[:length].all())
            self.assertEqual(len(starts(length)), len(set(starts(length))))

    def test_alpha_and_orientation(self):
        transparent = Image.new("RGBA", (3, 4), (0, 0, 0, 0))
        self.assertEqual(decode_image(png_bytes(transparent)).getpixel((0, 0)), (255, 255, 255))
        sample = Image.new("RGB", (3, 7), "red")
        exif = Image.Exif(); exif[274] = 6
        buffer = BytesIO(); sample.save(buffer, "JPEG", exif=exif)
        self.assertEqual(decode_image(buffer.getvalue()).size, (7, 3))

    def test_invalid_inputs(self):
        with self.assertRaises(Exception):
            decode_image(b"not an image")
        with self.assertRaises(ValueError):
            decode_image(png_bytes(Image.new("RGB", (8193, 1))))
        buffer = BytesIO(); Image.new("RGB", (4, 4)).save(buffer, "GIF")
        with self.assertRaises(ValueError):
            decode_image(buffer.getvalue())

    def test_perturbations_preserve_grid(self):
        source = make_document(1000)
        for name in CONDITIONS:
            a, b = degrade(source, name), degrade(source, name)
            self.assertEqual(a.size, source.size)
            np.testing.assert_array_equal(np.array(a), np.array(b))
        np.testing.assert_array_equal(np.array(degrade(source, "clean")), np.array(source))
        self.assertFalse(np.array_equal(np.array(source), np.array(degrade(source, "jpeg70"))))

    def test_augmentation_preserves_mask(self):
        class Base:
            def __len__(self): return 1
            def __getitem__(self, index):
                return {"image": torch.rand(3, 32, 32), "mask": torch.ones(1, 32, 32), "index": 10}
        augmented = JpegTrainingDataset(Base())
        for _ in range(12):
            sample = augmented[0]
            self.assertEqual(sample["index"], 10)
            self.assertTrue(torch.all(sample["mask"] == 1))
            self.assertEqual(tuple(sample["image"].shape), (3, 32, 32))

    def test_ela_shape_and_dtype(self):
        source = make_document(2024)
        result = ela(source)
        self.assertEqual(result.shape, (source.height, source.width, 3))
        self.assertEqual(result.dtype, np.uint8)
        # Clean PNG re-saved as JPEG should produce some non-zero residual.
        self.assertTrue(result.max() > 0)
        # ELA output must be strictly within [0, 255].
        self.assertGreaterEqual(int(result.min()), 0)
        self.assertLessEqual(int(result.max()), 255)

    def test_ela_deterministic(self):
        source = make_document(777)
        a, b = ela(source), ela(source)
        np.testing.assert_array_equal(a, b)

    def test_probe_conditions_exported(self):
        self.assertIn("jpeg70", PROBE_CONDITIONS)
        self.assertIn("jpeg90", PROBE_CONDITIONS)
        self.assertIn("jpeg50", PROBE_CONDITIONS)
        self.assertIn("blur07", PROBE_CONDITIONS)

    def test_regions_ranking_and_shape(self):
        prob = np.zeros((64, 64), dtype=np.float32)
        prob[5:15, 5:25] = 0.8    # 10×20 = 200 px (largest)
        prob[40:50, 40:55] = 0.9  # 10×15 = 150 px
        regions = summarize_regions(prob, threshold=0.5)
        self.assertEqual(len(regions), 2)
        self.assertEqual(regions[0]["rank"], 1)
        self.assertGreaterEqual(regions[0]["area_px"], regions[1]["area_px"])
        self.assertAlmostEqual(regions[0]["mean_confidence"], 0.8, places=3)
        for r in regions:
            self.assertIn("bbox", r)
            self.assertEqual(len(r["bbox"]), 4)

    def test_regions_empty_mask(self):
        prob = np.zeros((32, 32), dtype=np.float32)
        self.assertEqual(summarize_regions(prob, threshold=0.5), [])

    def test_regions_max_regions_capped(self):
        prob = np.zeros((128, 128), dtype=np.float32)
        # Place 15 isolated single pixels above threshold.
        for i in range(15):
            prob[i*8, i*8] = 0.9
        regions = summarize_regions(prob, threshold=0.5, max_regions=10)
        self.assertLessEqual(len(regions), 10)


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import socket
        class FakePredictor:
            metadata = {"model": "test", "device": "cpu"}
            def analyze(self, image, probe):
                probability = np.full((image.height, image.width), .6, dtype=np.float32)
                alt = probability if probe else None
                worst = probability if probe else None
                ela_arr = np.zeros((image.height, image.width, 3), dtype=np.uint8)
                return probability, alt, worst, ela_arr, {"test": True, "regions": [], "multi_probe": None, "probe": None}
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
        cls.port = cls.server.server_port
        cls.server.RequestHandlerClass = make_handler(FakePredictor(), cls.port)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.port}"
        # Verify clients can actually reach the server in this environment.
        try:
            probe = socket.create_connection(("127.0.0.1", cls.port), timeout=2)
            probe.close()
        except OSError as exc:
            cls.server.shutdown(); cls.server.server_close(); cls.thread.join()
            raise unittest.SkipTest(f"Loopback TCP connect blocked in this environment: {exc}") from exc

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join()

    def test_local_analysis(self):
        request = Request(self.url+"/api/analyze", data=png_bytes(Image.new("RGB", (9, 7))), headers={"Origin": self.url, "X-DocTrace-Request": "1"})
        with urlopen(request) as response:
            body = json.load(response)
            self.assertTrue(body["report"]["test"])
            self.assertIsNotNone(body["alternative"])
            self.assertIsNotNone(body["worst_case"])
            self.assertIsNotNone(body["ela"])
            self.assertIn("frame-ancestors 'none'", response.headers["Content-Security-Policy"])

    def test_rejects_foreign_origin(self):
        request = Request(self.url+"/api/analyze", data=b"test", headers={"Origin":"https://example.com", "X-DocTrace-Request":"1"})
        with self.assertRaises(HTTPError) as error:
            urlopen(request)
        self.assertEqual(error.exception.code, 403)

    def test_no_arbitrary_file_route(self):
        with self.assertRaises(HTTPError) as error:
            urlopen(self.url+"/../requirements.txt")
        self.assertEqual(error.exception.code, 404)

    def test_rejects_invalid_image(self):
        request = Request(self.url+"/api/analyze", data=b"bad image", headers={"Origin":self.url, "X-DocTrace-Request":"1"})
        with self.assertRaises(HTTPError) as error:
            urlopen(request)
        self.assertEqual(error.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
