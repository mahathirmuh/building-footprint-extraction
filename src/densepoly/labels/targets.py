"""All training targets of one tile, derived from its instance map after the geometric augmentation, so that they
always match the transformed tile (docs/METHOD.md §3.5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from densepoly.labels.boundary_weight import edge_weight
from densepoly.labels.contact import class_map
from densepoly.labels.orientation import orientation_targets


@dataclass
class Targets:
    classes: np.ndarray  # (H, W) uint8: 0 background, 1 building, 2 contact
    sin4: np.ndarray  # (H, W) float32, sin 4θ on valid pixels
    cos4: np.ndarray  # (H, W) float32, cos 4θ on valid pixels
    valid: np.ndarray  # (H, W) bool, where the θ loss applies (V)
    weight: np.ndarray  # (H, W) float32, boundary weight w


def derive_targets(
    instance_map: np.ndarray, contact_radius: int, theta_min_area: int, alpha: float, sigma: float
) -> Targets:
    orientation = orientation_targets(instance_map, theta_min_area)
    return Targets(
        classes=class_map(instance_map, contact_radius),
        sin4=orientation.sin4,
        cos4=orientation.cos4,
        valid=orientation.valid,
        weight=edge_weight(instance_map, alpha, sigma),
    )


def targets_from_config(instance_map: np.ndarray, labels) -> Targets:
    """derive_targets with the labels section of an experiment config (configs/base.yaml)."""
    return derive_targets(
        instance_map,
        int(labels.contact_radius_px),
        int(labels.theta_min_area_px),
        float(labels.edge_weight_alpha),
        float(labels.edge_weight_sigma_px),
    )
