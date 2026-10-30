"""Polygons of an instance map (docs/METHOD.md §6.4, §7) in our prediction format (docs/EVALUATION.md §1).

One exterior ring per instance, by mode:
- raw (B1-B4): the contour simplified by Douglas-Peucker with a fixed ε (1 px);
- mar (B0, B5, B5′): regularised along the minAreaRect θ of the instance pixels, the rule of the θ label (§3.3), with
  kappa null;
- theta (FULL, FULL+Y): regularised along the θ of the head, its weighted circular mean over the instance (§6.4),
  with κ.

Every ring is then clipped to the tile (a turned box can stick out of it) and rounded to 2 decimals. Rounding can break
a valid polygon, for instance a thin part that ends up crossing itself; such a ring falls back to the contour, whose
coordinates are multiples of 0.5 and survive the rounding (§7, clarified 30 Oct 2026).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from shapely import affinity
from shapely.geometry import Polygon, box

from densepoly.eval import coco_io
from densepoly.labels.orientation import pixel_theta
from densepoly.postproc.contours import mask_to_exterior
from densepoly.postproc.regularize import RegularizeParams, largest_polygon, outline_polygon, regularize, vertices

MODES = ("raw", "mar", "theta")
DECIMALS = 2


@dataclass(frozen=True)
class PolygonParams:
    mode: str  # raw | mar | theta
    raw_eps_px: float = 1.0
    regularize: RegularizeParams = field(default_factory=RegularizeParams)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"unknown polygon mode {self.mode!r}; expected one of {MODES}")

    @classmethod
    def from_config(cls, postproc) -> PolygonParams:
        """From the postproc section of an experiment config (configs/base.yaml)."""
        return cls(
            mode=str(postproc.polygon),
            raw_eps_px=float(postproc.raw_eps_px),
            regularize=RegularizeParams(
                rect_iou=float(postproc.rect_iou),
                dp_k=float(postproc.dp_k),
                snap_deg=float(postproc.snap_deg),
                oblique_min_px=float(postproc.oblique_min_px),
                edge_min_px=float(postproc.edge_min_px),
            ),
        )


@dataclass(frozen=True)
class InstancePolygon:
    instance: int  # value in the instance map
    points: np.ndarray  # (N, 2) exterior ring (x, y) in the tile frame, no repeated closing vertex
    path: str  # raw | box | snap | dp | contour
    fallback: bool
    theta_deg: float | None
    kappa: float | None


def head_theta(
    instance_map: np.ndarray, sin4: np.ndarray, cos4: np.ndarray, weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """θ (radians, [0, π/2)) and κ ∈ [0, 1] of instances 1..n from the head (§6.4). The two tanh outputs of a pixel are
    no unit vector, so each is normalised first; then the mean weighted by ω = 1 − P_bg gives
    θ = ¼·atan2(Σ ω u_s, Σ ω u_c) mod 90° and κ = ‖Σ ω u‖ / Σ ω."""
    norm = np.hypot(sin4, cos4)
    scale = np.where(norm > 0, weight / np.maximum(norm, 1e-12), 0.0)  # a zero vector has no direction
    n = int(instance_map.max())
    labels = instance_map.ravel()
    sums = [np.bincount(labels, weights=v.ravel(), minlength=n + 1)[1:] for v in (scale * sin4, scale * cos4)]
    total = np.bincount(labels, weights=np.where(norm > 0, weight, 0.0).ravel(), minlength=n + 1)[1:]
    theta = (np.arctan2(sums[0], sums[1]) / 4) % (np.pi / 2)
    kappa = np.hypot(*sums) / np.maximum(total, 1e-12)
    return theta, kappa


def tile_ring(polygon: Polygon, width: int, height: int) -> np.ndarray | None:
    """Ring of a polygon clipped to the tile and rounded to DECIMALS; None when it is no longer a valid polygon."""
    clipped = largest_polygon(polygon.intersection(box(0, 0, width, height)))
    if clipped is None:
        return None
    points = np.round(vertices(clipped), DECIMALS)
    points = points[np.any(points != np.roll(points, 1, axis=0), axis=1)]  # vertices the rounding merged
    if len(np.unique(points, axis=0)) < 3:
        return None
    ring = Polygon(points)
    return points if ring.is_valid and ring.area > 0 else None


def polygonize(
    instance_map: np.ndarray,
    params: PolygonParams,
    theta: np.ndarray | None = None,
    kappa: np.ndarray | None = None,
) -> list[InstancePolygon]:
    """One polygon per instance 1..n of the map, in instance order. Mode theta needs theta (radians) per instance:
    the head's (head_theta) or a given one, such as the θ of the GT polygons in the oracle; kappa is optional."""
    if params.mode == "theta" and theta is None:
        raise ValueError("mode theta needs a θ per instance")
    height, width = instance_map.shape
    out = []
    for value, window in enumerate(ndi.find_objects(instance_map), start=1):
        if window is None:
            continue
        mask = instance_map[window] == value
        x0, y0 = window[1].start, window[0].start
        contour = mask_to_exterior(mask)
        outline = outline_polygon(contour)
        theta_deg = k = None
        if params.mode == "raw":
            path, polygon = "raw", outline.simplify(params.raw_eps_px, preserve_topology=True)
        else:
            if params.mode == "mar":
                rows, cols = np.nonzero(mask)
                th = pixel_theta(rows, cols)
            else:
                th = float(theta[value - 1])
                k = None if kappa is None else float(kappa[value - 1])
            result = regularize(mask, th, params.regularize, contour)
            path, polygon, theta_deg = result.path, result.polygon, float(np.degrees(th))
        points = tile_ring(affinity.translate(polygon, x0, y0), width, height)
        if points is None:
            path, points = "contour", vertices(outline) + (x0, y0)  # multiples of 0.5, inside the tile
        out.append(InstancePolygon(value, points, path, path in ("dp", "contour"), theta_deg, k))
    return out


def to_results(polygons: list[InstancePolygon], scores: np.ndarray, image_id: int, category_id: int) -> list[dict]:
    """COCO results of one image (docs/EVALUATION.md §1); scores[i] is the score of instance i + 1."""
    return [
        coco_io.to_result(
            image_id, p.points, float(scores[p.instance - 1]), category_id, p.theta_deg, p.kappa, fallback=p.fallback
        )
        for p in polygons
    ]
