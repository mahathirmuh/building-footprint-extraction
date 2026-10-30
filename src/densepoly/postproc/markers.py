"""Foreground and markers of a three-class prediction (docs/METHOD.md §6.2, steps 2-3).

Every component labelling on the prediction side uses 4-connectivity, like the marching-squares contours.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

FOUR = ndi.generate_binary_structure(2, 1)


def label4(mask: np.ndarray) -> np.ndarray:
    """(H, W) int32 labels of the 4-connected components of a binary mask, 0 = background."""
    labels, _ = ndi.label(mask, structure=FOUR)
    return labels.astype(np.int32)


def drop_small(labels: np.ndarray, min_px: int) -> np.ndarray:
    """Labels with every label of fewer than min_px pixels removed, the others renumbered 1..n in their order."""
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_px
    keep[0] = False
    new_ids = np.zeros(sizes.size, np.int32)
    new_ids[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return new_ids[labels]


def foreground(p_bg: np.ndarray) -> np.ndarray:
    """F = (1 − P_bg) > 0.5: building or contact."""
    return (1.0 - p_bg) > 0.5


def markers(p_b: np.ndarray, p_k: np.ndarray, fg: np.ndarray, tau: float, min_px: int) -> np.ndarray:
    """Labelled markers M = (P_b − P_k > τ) & F, without components under min_px (m_min) pixels.

    Cutting to F comes before the labelling: otherwise one marker component could bridge two components of F, and one
    label would become an instance in two parts."""
    return drop_small(label4((p_b - p_k > tau) & fg), min_px)
