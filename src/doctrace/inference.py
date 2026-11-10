"""Bounded, native-scale tiled inference for local document inspection."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import time
import warnings

import numpy as np
from PIL import Image, ImageOps
import torch

from .degradations import degrade, ela, to_tensor, PROBE_CONDITIONS
from .model import build_model
from .regions import summarize_regions

TILE = 512
STRIDE = 384
MAX_PIXELS = 8_000_000
MAX_BYTES = 24 * 1024 * 1024


def decode_image(data):
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Choose an image smaller than 24 MB.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(BytesIO(data)) as source:
            if source.format not in ("PNG", "JPEG", "WEBP") or getattr(source, "n_frames", 1) != 1:
                raise ValueError("Choose a single PNG, JPEG, or WebP image.")
            if source.width * source.height > MAX_PIXELS or max(source.size) > 8192:
                raise ValueError("Image is too large. Use at most 8 megapixels and 8192 pixels per side.")
            oriented = ImageOps.exif_transpose(source)
            if "A" in oriented.getbands() or "transparency" in oriented.info:
                rgba = oriented.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                return Image.alpha_composite(background, rgba).convert("RGB")
            return oriented.convert("RGB")


def starts(length, tile=TILE, stride=STRIDE):
    if length < 1 or not 0 < stride <= tile:
        raise ValueError("Invalid tiling dimensions")
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


def tile_count(size):
    return len(starts(size[0])) * len(starts(size[1]))


def png_bytes(array):
    buffer = BytesIO()
    image = array if isinstance(array, Image.Image) else Image.fromarray(array)
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def overlay(image, probability, threshold=.5):
    rgb = np.asarray(image, dtype=np.float32).copy()
    active = probability >= threshold
    rgb[active] = .55 * rgb[active] + .45 * np.array([242, 90, 62])
    return np.clip(rgb, 0, 255).astype(np.uint8)


class Predictor:
    def __init__(self, checkpoint, device="auto"):
        path = Path(checkpoint)
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device)
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.model = build_model(state["config"].get("model", "small"))
        self.model.load_state_dict(state["model"], strict=True)
        self.model.to(self.device).eval()
        self.metadata = {"model": state["config"].get("architecture", "SmallUNet"),
                         "epoch": state["epoch"], "checkpoint_sha256": sha256(path.read_bytes()).hexdigest(),
                         "device": self.device.type, "tile_size": TILE, "stride": STRIDE}

    @torch.inference_mode()
    def predict(self, image):
        width, height = image.size
        if width * height > MAX_PIXELS or max(image.size) > 8192 or tile_count(image.size) > 64:
            raise ValueError("Image exceeds the local demo's size limit. Use a smaller image or a crop.")
        rgb = np.array(image.convert("RGB"))
        padded = np.pad(rgb, ((0, max(0, TILE-height)), (0, max(0, TILE-width)), (0, 0)), mode="edge")
        total = np.zeros(padded.shape[:2], dtype=np.float32)
        weights = np.zeros_like(total)
        # Raised-cosine weighting reduces tile seams; the floor keeps page edges covered.
        axis = np.maximum(np.hanning(TILE).astype(np.float32), .05)
        window = axis[:, None] * axis[None, :]
        positions = [(y, x) for y in starts(height) for x in starts(width)]
        for offset in range(0, len(positions), 4):
            batch_positions = positions[offset:offset+4]
            batch = torch.stack([to_tensor(Image.fromarray(padded[y:y+TILE, x:x+TILE])) for y, x in batch_positions])
            probabilities = self.model(batch.to(self.device)).sigmoid()[:, 0].cpu().numpy()
            if not np.isfinite(probabilities).all():
                raise RuntimeError("The model produced nonfinite values.")
            for (y, x), probability in zip(batch_positions, probabilities):
                total[y:y+TILE, x:x+TILE] += probability * window
                weights[y:y+TILE, x:x+TILE] += window
        return (total / weights)[:height, :width]

    def analyze(self, image, probe=True, threshold=.5):
        if not 0 < threshold < 1:
            raise ValueError("Threshold must be between 0 and 1.")
        started = time.perf_counter()
        probability = self.predict(image)

        # --- Multi-condition sensitivity probe ---
        probe_results = {}
        worst_case = None
        probe_maps = {}
        if probe:
            for condition in PROBE_CONDITIONS:
                probe_maps[condition] = self.predict(degrade(image, condition))
            # Per-condition absolute change and threshold flips vs clean prediction.
            for condition, alt_map in probe_maps.items():
                abs_change = np.abs(probability - alt_map)
                probe_results[condition] = {
                    "mean_absolute_change": float(abs_change.mean()),
                    "threshold_flip_fraction": float(((probability >= threshold) != (alt_map >= threshold)).mean()),
                }
            # Worst-case map: maximum absolute change across all probe conditions.
            worst_case = np.max(np.stack([np.abs(probability - pm) for pm in probe_maps.values()]), axis=0)

        # --- Error Level Analysis ---
        ela_map = ela(image)

        # --- Region analysis ---
        regions = summarize_regions(probability, threshold)

        # Backward-compatible single-probe field (jpeg70).
        alt_jpeg70 = probe_maps.get("jpeg70") if probe else None

        report = {
            "schema_version": 2, **self.metadata,
            "width": image.width, "height": image.height,
            "tiles": tile_count(image.size), "threshold": threshold,
            "flagged_pixel_fraction": float((probability >= threshold).mean()),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "probe": None,
            "multi_probe": None,
            "regions": regions,
            "interpretation": "Localization research prototype. Flagged pixels are model responses, not proof of alteration. No image-level authenticity verdict or calibrated confidence.",
            "limitations": "Real authentic-document false alarms and physical recapture are unvalidated. Tiled full-page inference is not the 512x512 benchmark protocol.",
        }
        if probe:
            report["probe"] = {
                "condition": "jpeg70",
                "mean_absolute_probability_change": float(np.abs(probability - alt_jpeg70).mean()),
                "threshold_flip_fraction": probe_results["jpeg70"]["threshold_flip_fraction"],
                "interpretation": "Response change after JPEG recompression; not calibrated uncertainty or evidence of tampering.",
            }
            report["multi_probe"] = {
                "conditions": probe_results,
                "worst_case_mean_change": float(worst_case.mean()),
                "worst_case_flip_fraction": float((worst_case >= 0.1).mean()),
                "interpretation": "Per-condition and aggregate sensitivity to jpeg90/70/50 and Gaussian blur. High values indicate unstable predictions, not confirmed tampering.",
            }
        return probability, alt_jpeg70, worst_case, ela_map, report
