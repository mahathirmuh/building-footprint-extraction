"""Vertex F1 (docs/EVALUATION.md §2.1, secondary metric): on a matched pair, the vertices of both polygons are paired
one-to-one by a Hungarian assignment on their distances, and a vertex pair counts when it is at most 2 px apart.
F1 pools the counts of all pairs: 2·Σ matches / (Σ GT vertices + Σ predicted vertices)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

MAX_DISTANCE_PX = 2.0


def ring_vertices(ring) -> np.ndarray:
    """(N, 2) vertices of a flat COCO ring, without the repeated closing vertex."""
    xy = np.reshape(np.asarray(ring, float), (-1, 2))
    if len(xy) > 1 and np.array_equal(xy[0], xy[-1]):
        xy = xy[:-1]
    return xy


def vertex_matches(gt: np.ndarray, pred: np.ndarray, max_distance: float = MAX_DISTANCE_PX) -> int:
    """Largest number of one-to-one vertex pairs at most max_distance apart (ties broken by total distance)."""
    if len(gt) == 0 or len(pred) == 0:
        return 0
    distance = cdist(gt, pred)
    feasible = distance <= max_distance
    # Any infeasible pair costs more than all feasible pairs together, so the assignment maximises their number.
    penalty = max_distance * min(distance.shape) + 1.0
    rows, cols = linear_sum_assignment(np.where(feasible, distance, penalty))
    return int(feasible[rows, cols].sum())
