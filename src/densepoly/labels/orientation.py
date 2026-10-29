"""Orientation θ (docs/METHOD.md §3.3, contribution N1): the main side direction of every building, from the rectangle
of cv2.minAreaRect, encoded as (sin 4θ, cos 4θ) because a building looks the same every 90°.

θ comes from a side vector of cv2.boxPoints, never from the angle cv2.minAreaRect returns, whose convention changed
between OpenCV versions. Both sides at a corner give the same 4θ; the longer one is used so that a degenerate box
(a building one pixel wide, e.g. cut by the tile border) still yields its direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi


def pixel_theta(rows: np.ndarray, cols: np.ndarray) -> float:
    """θ in [0, π/2) of a set of pixels, in image coordinates (x = column, y = row pointing down)."""
    points = np.column_stack([cols, rows]).astype(np.float32)
    box = cv2.boxPoints(cv2.minAreaRect(points))
    side_a, side_b = box[1] - box[0], box[2] - box[1]
    ex, ey = side_a if np.hypot(*side_a) >= np.hypot(*side_b) else side_b
    return float(np.arctan2(ey, ex) % (np.pi / 2))


def instance_pixels(instance_map: np.ndarray):
    """Yield (value, rows, cols) for every non-empty instance, in increasing value."""
    for value, box in enumerate(ndi.find_objects(instance_map), start=1):
        if box is None:
            continue
        rows, cols = np.nonzero(instance_map[box] == value)
        yield value, rows + box[0].start, cols + box[1].start


@dataclass
class OrientationTargets:
    sin4: np.ndarray  # (H, W) float32, sin 4θ on valid pixels, 0 elsewhere
    cos4: np.ndarray  # (H, W) float32, cos 4θ on valid pixels, 0 elsewhere
    valid: np.ndarray  # (H, W) bool, pixels of instances with at least min_area pixels (V)
    theta: dict[int, float]  # θ in radians per valid instance value


def orientation_targets(instance_map: np.ndarray, min_area: int) -> OrientationTargets:
    """Per-pixel (sin 4θ, cos 4θ) of every instance with at least min_area pixels (A_θ)."""
    sin4 = np.zeros(instance_map.shape, np.float32)
    cos4 = np.zeros(instance_map.shape, np.float32)
    valid = np.zeros(instance_map.shape, bool)
    theta = {}
    for value, rows, cols in instance_pixels(instance_map):
        if rows.size < min_area:
            continue
        theta[value] = pixel_theta(rows, cols)
        sin4[rows, cols] = np.sin(4 * theta[value])
        cos4[rows, cols] = np.cos(4 * theta[value])
        valid[rows, cols] = True
    return OrientationTargets(sin4, cos4, valid, theta)
