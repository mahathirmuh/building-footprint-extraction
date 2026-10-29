"""Morphological Building Index (docs/METHOD.md §2.1-2.2; Huang & Zhang 2011): bright structures about the size of
a house, from white top-hats by reconstruction with linear structuring elements in four directions.

b = max(R, G, B) / 255. For every direction d and scale s: WTH(s, d) = b − γ_R(b; s, d), where the opening by
reconstruction γ_R(b; s, d) = reconstruction(seed = erosion(b, line(s, d)), mask = b). DMP(s, d) =
|WTH(s + Δs, d) − WTH(s, d)| over consecutive scales, and MBI = the mean of all DMPs.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from skimage.draw import line
from skimage.morphology import reconstruction

ANGLES = (0, 45, 90, 135)  # degrees, counter-clockwise on screen from the x axis


def brightness(image: np.ndarray) -> np.ndarray:
    """(H, W) float32 in [0, 1]: the maximum of the visible bands (Huang & Zhang 2011)."""
    return image.max(axis=2).astype(np.float32) / 255.0


def line_footprint(length: int, angle: float) -> np.ndarray:
    """length × length uint8 footprint holding a line through the centre, about `length` px long (Euclidean) in the
    given direction, like MATLAB's strel('line'). The length must be odd so that the line is centred."""
    if length < 1 or length % 2 == 0:
        raise ValueError(f"line length must be a positive odd number, got {length}")
    centre = length // 2
    t = np.radians(angle)
    dr, dc = int(round(-centre * np.sin(t))), int(round(centre * np.cos(t)))  # rows grow downwards
    footprint = np.zeros((length, length), np.uint8)
    rows, cols = line(centre - dr, centre - dc, centre + dr, centre + dc)
    footprint[rows, cols] = 1
    return footprint


def opening_by_reconstruction(b: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """γ_R: erode with the footprint, then rebuild by geodesic dilation under b, so that every bright structure the
    erosion did not remove completely comes back with its exact shape."""
    seed = cv2.erode(b, footprint)  # the border counts as +inf, so it never erodes the tile edge
    return reconstruction(seed, b, method="dilation").astype(np.float32)


def mbi(b: np.ndarray, scales: Sequence[int], angles: Sequence[float] = ANGLES) -> np.ndarray:
    """(H, W) float32 MBI of a brightness image over the given increasing scales (px)."""
    if len(scales) < 2:
        raise ValueError("MBI needs at least two scales")
    total = np.zeros_like(b, dtype=np.float32)
    count = 0
    for angle in angles:
        previous = None
        for scale in scales:
            tophat = b - opening_by_reconstruction(b, line_footprint(scale, angle))
            if previous is not None:
                total += np.abs(tophat - previous)
                count += 1
            previous = tophat
    return total / count
