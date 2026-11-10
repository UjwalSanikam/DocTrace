"""Connected-component analysis on thresholded probability maps.

Turns a continuous probability array into a ranked list of suspicious regions.
No external dependencies beyond NumPy; uses a simple two-pass flood-fill via
the standard-library approach so scipy is not required.
"""
from __future__ import annotations

import numpy as np


def _label(binary: np.ndarray) -> tuple[np.ndarray, int]:
    """Two-pass connected-component labelling (4-connectivity) in pure NumPy.

    Returns (label_array, num_labels) where label_array is int32 with values
    1..num_labels for each connected component and 0 for background.
    """
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent: list[int] = [0]  # index 0 = background sentinel

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # First pass: provisional labels.
    next_label = 1
    for y in range(h):
        for x in range(w):
            if not binary[y, x]:
                continue
            up = labels[y - 1, x] if y > 0 else 0
            left = labels[y, x - 1] if x > 0 else 0
            if up == 0 and left == 0:
                parent.append(next_label)
                labels[y, x] = next_label
                next_label += 1
            elif up == 0:
                labels[y, x] = find(left)
            elif left == 0:
                labels[y, x] = find(up)
            else:
                labels[y, x] = find(up)
                union(up, left)

    # Second pass: flatten equivalences.
    remap: dict[int, int] = {}
    new_next = 1
    for y in range(h):
        for x in range(w):
            if labels[y, x] == 0:
                continue
            root = find(labels[y, x])
            if root not in remap:
                remap[root] = new_next
                new_next += 1
            labels[y, x] = remap[root]

    return labels, new_next - 1


def summarize_regions(probability: np.ndarray, threshold: float = 0.5, max_regions: int = 10) -> list[dict]:
    """Return a ranked list of suspicious connected regions above *threshold*.

    Each entry contains:
    - ``rank``: 1-based rank by area (largest first).
    - ``area_px``: number of pixels in the region.
    - ``area_fraction``: fraction of the full image.
    - ``mean_confidence``: mean model probability inside the region.
    - ``bbox``: ``[y_min, x_min, y_max, x_max]`` pixel bounding box (inclusive).
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0, 1)")
    binary = probability >= threshold
    if not binary.any():
        return []
    labels, num_labels = _label(binary)
    total_pixels = probability.size
    regions: list[dict] = []
    for label_id in range(1, num_labels + 1):
        mask = labels == label_id
        ys, xs = np.where(mask)
        area_px = int(mask.sum())
        regions.append({
            "rank": 0,  # filled in after sorting
            "area_px": area_px,
            "area_fraction": round(area_px / total_pixels, 6),
            "mean_confidence": round(float(probability[mask].mean()), 4),
            "bbox": [int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())],
        })
    regions.sort(key=lambda r: r["area_px"], reverse=True)
    for i, region in enumerate(regions[:max_regions], start=1):
        region["rank"] = i
    return regions[:max_regions]
