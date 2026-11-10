"""Efficient connected-component analysis on thresholded document probability maps.

Extracts 4-connected components from a continuous 2D probability map and aggregates
area, mean model response, and bounding box coordinates in a single vectorized pass
without O(N * K) full-image rescans.
"""
from __future__ import annotations

from typing import Any, List, Optional
import numpy as np


class RegionList(list):
    """List of region dictionaries carrying collection-level metadata.

    Subclasses built-in list so callers treating the result as a standard list of
    regions (length, indexing, JSON serialization) remain fully backward-compatible,
    while also exposing total region count and truncation status.
    """

    def __init__(
        self,
        regions: List[dict[str, Any]],
        total_count: int,
        returned_count: int,
        truncated: bool,
        threshold: float,
    ):
        super().__init__(regions)
        self.total_count = total_count
        self.returned_count = returned_count
        self.truncated = truncated
        self.threshold = threshold

    def to_dict(self) -> dict[str, Any]:
        """Return the regions and metadata as a standard dictionary."""
        return {
            "threshold": self.threshold,
            "total_regions": self.total_count,
            "returned_regions": self.returned_count,
            "truncated": self.truncated,
            "regions": list(self),
        }


def _validate_inputs(
    probability: np.ndarray,
    threshold: float,
    max_regions: Optional[int],
) -> None:
    if not isinstance(probability, np.ndarray):
        raise TypeError(f"probability must be a numpy.ndarray, got {type(probability).__name__}")
    if probability.ndim != 2:
        raise ValueError(f"probability must be a 2D array, got shape {probability.shape}")
    h, w = probability.shape
    if h == 0 or w == 0:
        raise ValueError(f"probability array cannot have empty dimensions: shape={probability.shape}")
    if not np.all(np.isfinite(probability)):
        raise ValueError("probability array contains non-finite values (NaN or Inf)")
    if not isinstance(threshold, (int, float)):
        raise TypeError(f"threshold must be float, got {type(threshold).__name__}")
    if not (0.0 < float(threshold) < 1.0):
        raise ValueError(f"threshold must be in open interval (0, 1), got {threshold}")
    if max_regions is not None:
        if not isinstance(max_regions, (int, np.integer)):
            raise TypeError(f"max_regions must be int or None, got {type(max_regions).__name__}")
        if max_regions < 0:
            raise ValueError(f"max_regions must be non-negative, got {max_regions}")


def _label_4way(binary: np.ndarray) -> tuple[np.ndarray, int]:
    """Two-pass 4-connectivity connected-component labeling.

    Parameters
    ----------
    binary : np.ndarray
        2D boolean array where True represents active (tampered-response) pixels.

    Returns
    -------
    labels : np.ndarray
        int32 array with shape matching binary, where 0 is background and
        1..num_components are contiguous component labels.
    num_components : int
        Total number of disjoint 4-connected components.
    """
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent: list[int] = [0]
    next_label = 1

    for y in range(h):
        row_bin = binary[y]
        if not row_bin.any():
            continue
        for x in range(w):
            if not row_bin[x]:
                continue
            up = labels[y - 1, x] if y > 0 else 0
            left = labels[y, x - 1] if x > 0 else 0

            if up == 0 and left == 0:
                parent.append(next_label)
                labels[y, x] = next_label
                next_label += 1
            elif up == 0:
                root = left
                while parent[root] != root:
                    root = parent[root]
                labels[y, x] = root
            elif left == 0:
                root = up
                while parent[root] != root:
                    root = parent[root]
                labels[y, x] = root
            else:
                root_up = up
                while parent[root_up] != root_up:
                    root_up = parent[root_up]
                root_left = left
                while parent[root_left] != root_left:
                    root_left = parent[root_left]
                labels[y, x] = root_up
                if root_up != root_left:
                    parent[root_left] = root_up

    if next_label == 1:
        return labels, 0

    # Flatten disjoint-set parent array
    parent_arr = np.array(parent, dtype=np.int32)
    for i in range(1, len(parent_arr)):
        root = i
        while parent_arr[root] != root:
            root = parent_arr[root]
        parent_arr[i] = root

    # Remap roots to contiguous 1..num_components
    unique_roots = np.unique(parent_arr[1:])
    num_components = int(len(unique_roots))
    remap = np.zeros(len(parent_arr), dtype=np.int32)
    remap[unique_roots] = np.arange(1, num_components + 1, dtype=np.int32)
    final_map = remap[parent_arr]

    return final_map[labels], num_components


def summarize_regions(
    probability: np.ndarray,
    threshold: float = 0.5,
    max_regions: Optional[int] = 10,
) -> RegionList:
    """Return a deterministic ranked list of 4-connected regions above *threshold*.

    Parameters
    ----------
    probability : np.ndarray
        2D float array with values in [0, 1] representing per-pixel model responses.
    threshold : float, default 0.5
        Binarization cutoff in (0, 1). Pixels with probability >= threshold are marked active.
    max_regions : int or None, default 10
        Maximum number of largest regions to return. If None, all regions are returned.
        Total region count is always computed and recorded on the returned RegionList.

    Returns
    -------
    RegionList
        A list of region dictionaries sorted deterministically by area (descending) with
        tie-breaking by mean response (descending), y_min, x_min, y_max, x_max.
        Each dictionary contains:
        - ``rank``: 1-based rank among returned regions.
        - ``area_px``: number of active pixels in the component.
        - ``area_fraction``: fraction of image pixels occupied by the component.
        - ``mean_model_response``: mean raw model probability inside the component.
        - ``mean_confidence``: backward-compatible alias to ``mean_model_response``.
        - ``bbox``: ``[y_min, x_min, y_max, x_max]`` in inclusive integer pixel coordinates.

        The returned RegionList also carries:
        - ``total_count``: total number of components found in the mask.
        - ``returned_count``: number of components included in this list.
        - ``truncated``: True if total_count > returned_count.
        - ``threshold``: the threshold used for this extraction.
    """
    _validate_inputs(probability, threshold, max_regions)

    h, w = probability.shape
    total_pixels = h * w
    binary = probability >= float(threshold)

    if not binary.any():
        return RegionList([], total_count=0, returned_count=0, truncated=False, threshold=float(threshold))

    labels, num_components = _label_4way(binary)
    if num_components == 0:
        return RegionList([], total_count=0, returned_count=0, truncated=False, threshold=float(threshold))

    ys, xs = np.nonzero(labels)
    flat_labels = labels[ys, xs]
    flat_probs = probability[ys, xs].astype(np.float64)

    # Vectorized single-pass aggregation across all components:
    areas = np.bincount(flat_labels, minlength=num_components + 1)
    sums = np.bincount(flat_labels, weights=flat_probs, minlength=num_components + 1)

    y_min = np.full(num_components + 1, h, dtype=np.int32)
    y_max = np.full(num_components + 1, -1, dtype=np.int32)
    x_min = np.full(num_components + 1, w, dtype=np.int32)
    x_max = np.full(num_components + 1, -1, dtype=np.int32)

    np.minimum.at(y_min, flat_labels, ys)
    np.maximum.at(y_max, flat_labels, ys)
    np.minimum.at(x_min, flat_labels, xs)
    np.maximum.at(x_max, flat_labels, xs)

    # Deterministic sorting order:
    # 1. area_px descending
    # 2. mean_model_response descending
    # 3. y_min ascending
    # 4. x_min ascending
    # 5. y_max ascending
    # 6. x_max ascending
    cids = np.arange(1, num_components + 1, dtype=np.int32)
    mean_responses = sums[cids] / np.maximum(areas[cids], 1)

    # Build tuples for stable sort
    sort_keys = [
        (
            -int(areas[cid]),
            -float(round(mean_responses[cid - 1], 6)),
            int(y_min[cid]),
            int(x_min[cid]),
            int(y_max[cid]),
            int(x_max[cid]),
            cid,
        )
        for cid in cids
    ]
    sort_keys.sort()

    limit = num_components if max_regions is None else min(max_regions, num_components)
    selected_cids = [key[-1] for key in sort_keys[:limit]]

    regions: list[dict[str, Any]] = []
    for rank, cid in enumerate(selected_cids, start=1):
        area_px = int(areas[cid])
        mean_resp = float(round(sums[cid] / area_px, 4))
        bbox = [int(y_min[cid]), int(x_min[cid]), int(y_max[cid]), int(x_max[cid])]
        regions.append({
            "rank": rank,
            "area_px": area_px,
            "area_fraction": float(round(area_px / total_pixels, 6)),
            "mean_model_response": mean_resp,
            "mean_confidence": mean_resp,  # backward compatibility alias
            "bbox": bbox,
        })

    truncated = num_components > len(regions)
    return RegionList(
        regions,
        total_count=num_components,
        returned_count=len(regions),
        truncated=truncated,
        threshold=float(threshold),
    )
