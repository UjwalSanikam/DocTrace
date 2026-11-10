"""Deterministic image perturbations that preserve the annotation grid."""
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
    """Error Level Analysis: re-save at fixed JPEG quality, return amplified abs residual.

    Returns a uint8 RGB array the same size as *image*. Regions with a different
    compression history (e.g., inserted from another source) tend to show higher
    residuals than authentic, multiply-compressed surroundings.

    *amplify* scales the absolute difference for display; pure pixel values are
    divided by amplify before clamping so that a residual of 255/amplify maps to
    full white. Default value of 15 gives visible contrast for typical documents.
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
