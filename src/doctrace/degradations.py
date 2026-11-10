"""Deterministic image perturbations and forensic inspection residuals."""
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter

CONDITIONS = ("clean", "jpeg90", "jpeg70", "jpeg50", "resize75", "resize50", "blur07")

# Probe conditions run during inference-time sensitivity analysis.
PROBE_CONDITIONS = ("jpeg90", "jpeg70", "jpeg50", "blur07")


def degrade(image, condition):
    image = image.convert("RGB")
    if condition == "clean":
        return image.copy()
    if condition in ("jpeg90", "jpeg70", "jpeg50"):
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=int(condition[4:]), subsampling=2)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB")
    if condition in ("resize75", "resize50"):
        scale = int(condition[6:]) / 100
        size = tuple(max(1, round(side * scale)) for side in image.size)
        return image.resize(size, Image.Resampling.BICUBIC).resize(image.size, Image.Resampling.BICUBIC)
    if condition == "blur07":
        return image.filter(ImageFilter.GaussianBlur(0.7))
    raise ValueError(f"Unknown condition: {condition}")


def ela(image, quality=90, amplify=15):
    """Error Level Analysis: re-save at fixed JPEG quality, return amplified absolute residual.

    Returns a uint8 RGB array the same size as *image*. Computes:
        np.clip(np.abs(image - jpeg_recompressed(image, quality)) * amplify, 0, 255)

    Interpretation limits:
    - ELA is an inspection heuristic visualizing recompression residuals, not an authenticity verdict.
    - Ordinary, unedited high-contrast text, lines, and sharp edges naturally produce strong residuals.
    - It does not establish tampering, splicing, or compression-history differences on its own.
    - Non-JPEG inputs (PNG, WebP) or previously resized/recompressed documents do not possess a pristine JPEG history.
    - *amplify* (default 15) scales residuals for visual contrast and clips values to [0, 255].
    """
    image = image.convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality, subsampling=2)
    buffer.seek(0)
    with Image.open(buffer) as recompressed:
        recompressed.load()
        diff = np.abs(np.array(image, dtype=np.float32) - np.array(recompressed, dtype=np.float32))
    return np.clip(diff * amplify, 0, 255).astype(np.uint8)


def to_tensor(image):
    import torch
    return torch.from_numpy(np.array(image.convert("RGB"), copy=True)).permute(2, 0, 1).float().div_(255)
