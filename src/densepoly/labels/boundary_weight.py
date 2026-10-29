"""Boundary weight (docs/METHOD.md §3.4, after U-Net and Bischke et al. 2019): w = 1 + α·exp(−d² / 2σ²), with d the
distance to the nearest boundary pixel. Padding pixels get w = 0 later, when the targets are padded."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def boundary_mask(instance_map: np.ndarray) -> np.ndarray:
    """Pixels whose label differs from at least one 4-neighbour: both sides of every edge, including the edges between
    a building and the background. The tile border itself is not a boundary."""
    boundary = np.zeros(instance_map.shape, bool)
    vertical = instance_map[1:, :] != instance_map[:-1, :]
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    horizontal = instance_map[:, 1:] != instance_map[:, :-1]
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    return boundary


def edge_weight(instance_map: np.ndarray, alpha: float, sigma: float) -> np.ndarray:
    """(H, W) float32 weight, 1 + α on boundary pixels and decaying to 1 away from them (1 everywhere without one)."""
    boundary = boundary_mask(instance_map)
    if not boundary.any():
        return np.ones(instance_map.shape, np.float32)
    distance = ndi.distance_transform_edt(~boundary)
    return (1.0 + alpha * np.exp(-(distance**2) / (2.0 * sigma**2))).astype(np.float32)
