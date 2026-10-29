"""Gap map (docs/METHOD.md §2.3, contribution N3): narrow dark lines between adjoining roofs, from black top-hats
BTH(d) = closing(b; line(L_g, d)) − b in four directions, with L_g longer than a gap and shorter than a roof.

Modes (one is chosen visually in Phase 2, then frozen):
- ``max``: max_d BTH(d), as in the ETS demo;
- ``contrast``: max_d BTH(d) − min_d BTH(d). Only the line perpendicular to a thin dark gap fills it, while every
  direction fills a round dark spot (tree shadow, vegetation texture), so isotropic noise cancels out.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from densepoly.priors.mbi import ANGLES, line_footprint

GAP_MODES = ("max", "contrast")


def black_tophats(b: np.ndarray, length: int, angles: Sequence[float] = ANGLES) -> np.ndarray:
    """(len(angles), H, W) float32 black top-hats of a brightness image."""
    return np.stack([cv2.morphologyEx(b, cv2.MORPH_BLACKHAT, line_footprint(length, a)) for a in angles])


def gap_map(b: np.ndarray, length: int, mode: str) -> np.ndarray:
    """(H, W) float32 gap map of a brightness image."""
    tophats = black_tophats(b, length)
    if mode == "max":
        return tophats.max(axis=0)
    if mode == "contrast":
        return tophats.max(axis=0) - tophats.min(axis=0)
    raise ValueError(f"unknown gap mode {mode!r}; expected one of {GAP_MODES}")
