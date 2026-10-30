"""Orientation error of matched pairs (docs/EVALUATION.md §2.1), a diagnostic of the θ head (N1).

θ_GT is the direction of a GT polygon under the rule of the orientation labels: the longer side of its
cv2.minAreaRect, modulo 90° (densepoly.labels.orientation). θ_pred is the theta_deg a method stores with each polygon,
i.e. the θ it regularised with. The error is the angle between both directions modulo 90°, in [0, 45].
"""

from __future__ import annotations

import math

import numpy as np

from densepoly.labels.orientation import points_theta


def polygon_theta_deg(ring) -> float:
    """θ in [0, 90) degrees of a flat COCO ring [x1, y1, x2, y2, ...]."""
    return math.degrees(points_theta(np.reshape(np.asarray(ring, float), (-1, 2))))


def theta_error_deg(theta_pred_deg: float, theta_gt_deg: float) -> float:
    """min(Δ mod 90, 90 − Δ mod 90) with Δ = |θ_pred − θ_GT|, in degrees."""
    delta = abs(theta_pred_deg - theta_gt_deg) % 90.0
    return min(delta, 90.0 - delta)
