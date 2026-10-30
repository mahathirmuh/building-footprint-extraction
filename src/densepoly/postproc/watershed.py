"""Instances of a prediction (docs/METHOD.md §6.2 steps 4-7 for three-class models, §6.3 for binary ones).

A_min is the smallest instance kept, applied after every instance has been split into its 4-connected parts; for binary
models it is the only post-processing parameter (docs/EXPERIMENTS.md §4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.measure import label
from skimage.segmentation import watershed

from densepoly.postproc.markers import drop_small, foreground, label4, markers


@dataclass(frozen=True)
class Instances:
    instance_map: np.ndarray  # (H, W) int32, 0 = background, instances 1..n
    scores: np.ndarray  # (n,) float, score of instance i + 1

    @property
    def count(self) -> int:
        return len(self.scores)


def split_parts(instance_map: np.ndarray, min_area: int) -> np.ndarray:
    """Every instance split into its 4-connected parts (each part a new instance), parts under min_area dropped."""
    parts = label(instance_map, background=0, connectivity=1).astype(np.int32)
    return drop_small(parts, min_area)


def mean_per_instance(instance_map: np.ndarray, values: np.ndarray) -> np.ndarray:
    """(n,) mean of values over the pixels of instances 1..n."""
    n = int(instance_map.max())
    sums = np.bincount(instance_map.ravel(), weights=values.ravel().astype(float), minlength=n + 1)
    counts = np.bincount(instance_map.ravel(), minlength=n + 1)
    return sums[1:] / np.maximum(counts[1:], 1)


def watershed_instances(landscape: np.ndarray, marker_labels: np.ndarray, fg: np.ndarray, min_area: int) -> np.ndarray:
    """Instance map: watershed of the landscape from the markers inside fg; every component of fg that no marker
    reaches becomes an instance of its own; then split_parts."""
    flooded = watershed(landscape, markers=marker_labels, mask=fg).astype(np.int32)  # connectivity 1
    unreached = label4(fg & (flooded == 0))
    flooded[unreached > 0] = unreached[unreached > 0] + flooded.max()
    return split_parts(flooded, min_area)


def three_class_instances(
    p_bg: np.ndarray, p_b: np.ndarray, p_k: np.ndarray, tau: float, marker_min_px: int, min_area: int
) -> Instances:
    """§6.2: F, markers, watershed of −P_b, unreached components, 4-connected parts; score = mean 1 − P_bg."""
    fg = foreground(p_bg)
    instance_map = watershed_instances(-p_b, markers(p_b, p_k, fg, tau, marker_min_px), fg, min_area)
    return Instances(instance_map, mean_per_instance(instance_map, 1.0 - p_bg))


def binary_instances(probability: np.ndarray, min_area: int) -> Instances:
    """§6.3: 4-connected components of probability > 0.5; score = mean probability."""
    instance_map = drop_small(label4(probability > 0.5), min_area)
    return Instances(instance_map, mean_per_instance(instance_map, probability))
