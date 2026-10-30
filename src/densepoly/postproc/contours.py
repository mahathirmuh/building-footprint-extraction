"""Exterior ring of an instance mask by marching squares (docs/METHOD.md §6.1).

At level 0.5 of the zero-padded mask the contour runs halfway between the centres of edge pixels and background pixels.
In continuous coordinates, where pixel (r, c) covers [c, c+1] × [r, r+1], that is the pixel border itself, so
rasterising the ring again with pycocotools gives the mask back: only the corners are cut, by 1/8 px² each, and no
pixel centre changes side. cv2.findContours traces the centres of the edge pixels instead; its polygons shrink by half a
pixel, which costs a 10 × 10 px house about 20% IoU. Holes are ignored, since COCO polygons cannot have them.
"""

from __future__ import annotations

import numpy as np
from skimage.measure import find_contours


def shoelace_area(points: np.ndarray) -> float:
    """Signed area of an (N, 2) ring, with or without a repeated closing vertex."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def mask_to_exterior(mask: np.ndarray) -> np.ndarray:
    """(N, 2) exterior ring (x, y) of a binary mask, without a repeated closing vertex: the largest contour of the
    padded mask at level 0.5, moved back by the padding and from pixel-centre to pixel-edge coordinates. The default
    fully_connected="low" of find_contours keeps diagonal foreground pixels apart, as the 4-connectivity of instances
    requires (docs/METHOD.md §6.2)."""
    if not mask.any():
        raise ValueError("the mask is empty")
    contours = find_contours(np.pad(mask.astype(np.uint8), 1), 0.5)
    ring = max(contours, key=lambda c: abs(shoelace_area(c)))  # (row, col); the order does not change |area|
    xy = ring[:, ::-1] - 1 + 0.5
    return xy[:-1] if np.array_equal(xy[0], xy[-1]) else xy
