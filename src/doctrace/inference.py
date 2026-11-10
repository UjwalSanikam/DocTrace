"""Bounded, native-scale tiled inference for local document inspection."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import time
import warnings
from typing import Optional

import numpy as np
from PIL import Image, ImageOps
import torch

from .degradations import degrade, ela, to_tensor, PROBE_CONDITIONS
from .model import build_model
from .regions import summarize_regions, RegionList

TILE = 512
STRIDE = 384
MAX_PIXELS = 8_000_000
MAX_BYTES = 24 * 1024 * 1024
RESPONSE_CHANGE_CUTOFF = 0.10


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
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
        )
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.model = build_model(state["config"].get("model", "small"))
        self.model.load_state_dict(state["model"], strict=True)
        self.model.to(self.device).eval()
        self.metadata = {
            "model": state["config"].get("architecture", "SmallUNet"),
            "epoch": state["epoch"],
            "checkpoint_sha256": sha256(path.read_bytes()).hexdigest(),
            "device": self.device.type,
            "tile_size": TILE,
            "stride": STRIDE,
        }

    @torch.inference_mode()
    def predict(self, image):
        width, height = image.size
        if width * height > MAX_PIXELS or max(image.size) > 8192 or tile_count(image.size) > 64:
            raise ValueError("Image exceeds the local demo's size limit. Use a smaller image or a crop.")
        rgb = np.array(image.convert("RGB"))
        padded = np.pad(rgb, ((0, max(0, TILE - height)), (0, max(0, TILE - width)), (0, 0)), mode="edge")
        total = np.zeros(padded.shape[:2], dtype=np.float32)
        weights = np.zeros_like(total)
        # Raised-cosine weighting reduces tile seams; the floor keeps page edges covered.
        axis = np.maximum(np.hanning(TILE).astype(np.float32), .05)
        window = axis[:, None] * axis[None, :]
        positions = [(y, x) for y in starts(height) for x in starts(width)]
        for offset in range(0, len(positions), 4):
            batch_positions = positions[offset:offset + 4]
            batch = torch.stack([to_tensor(Image.fromarray(padded[y:y + TILE, x:x + TILE])) for y, x in batch_positions])
            probabilities = self.model(batch.to(self.device)).sigmoid()[:, 0].cpu().numpy()
            if not np.isfinite(probabilities).all():
                raise RuntimeError("The model produced nonfinite values.")
            for (y, x), probability in zip(batch_positions, probabilities):
                total[y:y + TILE, x:x + TILE] += probability * window
                weights[y:y + TILE, x:x + TILE] += window
        return (total / weights)[:height, :width]

    def analyze(self, image, probe=True, threshold=.5):
        thresh = float(threshold)
        if not 0.0 < thresh < 1.0:
            raise ValueError(f"Threshold must be in (0, 1), got {threshold}")
        started = time.perf_counter()
        probability = self.predict(image)

        # --- Multi-condition sensitivity probe ---
        probe_results = {}
        worst_case = None
        probe_maps = {}
        if probe:
            for condition in PROBE_CONDITIONS:
                probe_maps[condition] = self.predict(degrade(image, condition))

            # Per-condition metrics:
            # - mean_absolute_change: mean(|p_clean - p_cond|)
            # - threshold_flip_fraction: mean((p_clean >= thresh) != (p_cond >= thresh))
            # - large_response_change_fraction: mean(|p_clean - p_cond| >= 0.10)
            flip_masks = []
            abs_change_maps = []
            for condition, alt_map in probe_maps.items():
                abs_change = np.abs(probability - alt_map)
                abs_change_maps.append(abs_change)
                flip_mask = (probability >= thresh) != (alt_map >= thresh)
                flip_masks.append(flip_mask)
                large_change_mask = abs_change >= RESPONSE_CHANGE_CUTOFF
                probe_results[condition] = {
                    "condition": condition,
                    "threshold": thresh,
                    "mean_absolute_change": float(abs_change.mean()),
                    "threshold_flip_fraction": float(flip_mask.mean()),
                    "response_change_cutoff": RESPONSE_CHANGE_CUTOFF,
                    "large_response_change_fraction": float(large_change_mask.mean()),
                }

            # Multi-condition aggregate metrics:
            worst_case = np.max(np.stack(abs_change_maps), axis=0)
            multi_flip_mask = np.any(np.stack(flip_masks), axis=0)
            multi_large_change_mask = worst_case >= RESPONSE_CHANGE_CUTOFF

            worst_case_mean_change = float(worst_case.mean())
            worst_case_flip_fraction = float(multi_flip_mask.mean())
            worst_case_large_response_fraction = float(multi_large_change_mask.mean())

        # --- Error Level Analysis ---
        ela_map = ela(image, quality=90, amplify=15)

        # --- Region analysis at the explicit analysis threshold ---
        regions: RegionList = summarize_regions(probability, threshold=thresh, max_regions=10)

        # Backward-compatible single-probe field (jpeg70).
        alt_jpeg70 = probe_maps.get("jpeg70") if probe else None

        report = {
            "schema_version": 3,
            **self.metadata,
            "width": image.width,
            "height": image.height,
            "tiles": tile_count(image.size),
            "threshold": thresh,
            "flagged_pixel_fraction": float((probability >= thresh).mean()),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "regions": list(regions),
            "total_regions_count": regions.total_count,
            "returned_regions_count": regions.returned_count,
            "regions_truncated": regions.truncated,
            "ela": {
                "quality": 90,
                "subsampling": "4:2:0",
                "amplification": 15,
                "clipping": [0, 255],
                "description": "Amplified absolute difference between input and a JPEG recompression at quality 90.",
                "interpretation": (
                    "Error Level Analysis is an inspection heuristic visualizing recompression residuals. "
                    "Ordinary unedited high-contrast text naturally produces strong residuals; "
                    "residual differences do not prove tampering or establish authenticity."
                ),
            },
            "probe": None,
            "multi_probe": None,
            "interpretation": (
                "Localization research prototype. Flagged pixels are raw model responses, "
                "not proof of alteration. No image-level authenticity verdict or calibrated fraud probability."
            ),
            "limitations": (
                "Real authentic-document false alarms and physical recapture are unvalidated. "
                "Tiled full-page inference is an engineering inspection tool, not the fixed-patch benchmark protocol."
            ),
        }

        if probe and alt_jpeg70 is not None:
            report["probe"] = {
                "condition": "jpeg70",
                "threshold": thresh,
                "mean_absolute_probability_change": float(np.abs(probability - alt_jpeg70).mean()),
                "threshold_flip_fraction": probe_results["jpeg70"]["threshold_flip_fraction"],
                "response_change_cutoff": RESPONSE_CHANGE_CUTOFF,
                "large_response_change_fraction": probe_results["jpeg70"]["large_response_change_fraction"],
                "interpretation": (
                    f"Response change after JPEG quality 70 recompression at threshold {thresh:.2f}; "
                    "sensitivity measure, not calibrated uncertainty or evidence of tampering."
                ),
            }
            report["multi_probe"] = {
                "threshold": thresh,
                "response_change_cutoff": RESPONSE_CHANGE_CUTOFF,
                "conditions_tested": list(PROBE_CONDITIONS),
                "conditions": probe_results,
                "worst_case_mean_change": worst_case_mean_change,
                "worst_case_threshold_flip_fraction": worst_case_flip_fraction,
                "worst_case_large_response_change_fraction": worst_case_large_response_fraction,
                # Backward compatibility alias
                "worst_case_flip_fraction": worst_case_flip_fraction,
                "large_response_change_fraction": worst_case_large_response_fraction,
                "interpretation": (
                    f"Sensitivity evaluated across {', '.join(PROBE_CONDITIONS)}. "
                    f"worst_case_threshold_flip_fraction is the fraction of pixels whose thresholded binary decision "
                    f"(at {thresh:.2f}) flips under any condition. "
                    f"worst_case_large_response_change_fraction is the fraction of pixels whose absolute response shift "
                    f"reaches or exceeds {RESPONSE_CHANGE_CUTOFF:.2f}."
                ),
            }

        return probability, alt_jpeg70, worst_case, ela_map, report
