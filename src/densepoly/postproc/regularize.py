"""Polygon regularisation of one instance along its orientation θ (docs/METHOD.md §7, steps 1-3).

1. Box path: the box of the second-order moments of the instance pixels, turned to θ. Its sides come from the pixel
   variance along and across θ, w = √(12·var + 1) (+1: the variance of n pixel centres is (n² − 1)/12), so one noisy
   bump on the border barely moves it, unlike a bounding box. It is the output when its IoU with the contour is ≥ ρ.
2. Snap path, otherwise: in the frame turned by −θ, Douglas-Peucker with ε = k·√A, then every edge within δ of
   horizontal or vertical becomes a line y = ȳ or x = x̄ (short oblique edges are raster steps or cut corners and join
   the nearest of the two), consecutive H or V edges merge, edges shorter than ℓ_min are dropped while more than four
   remain, and the new vertices are the intersections of consecutive lines.
3. A result must be valid, with an area, and keep IoU ≥ 0.7 with the contour; otherwise the DP polygon is tried, and
   last the contour itself.

Ported from reference/prototypes/regularize_proto.py (the reference for the expected vertex counts).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from densepoly.postproc.contours import mask_to_exterior

MIN_IOU = 0.7  # §7.3: the regularised polygon keeps at least this IoU with the contour
COLLINEAR_TOL = 1e-3  # px: vertices this close to the line of their neighbours are dropped after snapping


@dataclass(frozen=True)
class RegularizeParams:
    rect_iou: float = 0.90  # ρ: minimum IoU of the moment box with the contour
    dp_k: float = 0.04  # k: Douglas-Peucker tolerance ε = k·√A
    snap_deg: float = 15.0  # δ: tolerance of an H or V edge, in degrees
    oblique_min_px: float = 6.0  # ℓ_O = max(oblique_min_px, 4ε): shorter oblique edges join H or V
    edge_min_px: float = 2.0  # ℓ_min = max(edge_min_px, 2ε): shorter edges are dropped while > 4 remain


@dataclass(frozen=True)
class Regularized:
    polygon: Polygon
    path: str  # box | snap | dp | contour

    @property
    def fallback(self) -> bool:
        """The snap path failed its check (§7.3), so the DP polygon or the contour was kept."""
        return self.path in ("dp", "contour")


def rotate(points: np.ndarray, angle: float, center) -> np.ndarray:
    """Turn (x, y) points by angle (radians) about center; with y pointing down, a positive angle is clockwise on
    screen, so rotating by −θ brings a side of orientation θ onto the x axis."""
    c, s = np.cos(angle), np.sin(angle)
    return (np.asarray(points, float) - center) @ np.array([[c, s], [-s, c]]) + center


def vertices(polygon: Polygon) -> np.ndarray:
    """(N, 2) exterior vertices without the repeated closing vertex."""
    return np.asarray(polygon.exterior.coords)[:-1]


def largest_polygon(geometry: BaseGeometry) -> Polygon | None:
    """The largest polygon of a geometry (a make_valid or clip result can be a collection), None without one."""
    parts = [g for g in shapely.get_parts(geometry) if isinstance(g, Polygon) and g.area > 0]
    return max(parts, key=lambda g: g.area) if parts else None


def outline_polygon(ring: np.ndarray) -> Polygon:
    """The contour as a valid polygon (a marching-squares ring already is one; make_valid is a safety net)."""
    polygon = Polygon(ring)
    if not polygon.is_valid:
        polygon = largest_polygon(shapely.make_valid(polygon))
    if polygon is None or polygon.area <= 0:
        raise ValueError("the contour has no area")
    return polygon


def iou(a: Polygon, b: Polygon) -> float:
    union = a.union(b).area
    return a.intersection(b).area / union if union > 0 else 0.0


def moment_box(rows: np.ndarray, cols: np.ndarray, theta: float) -> Polygon:
    """Box of the second-order moments of the pixels (rows, cols), with sides along θ (§7.1)."""
    centres = np.column_stack([cols + 0.5, rows + 0.5]).astype(float)
    center = centres.mean(axis=0)
    aligned = rotate(centres, -theta, center)
    w, h = np.sqrt(12 * aligned.var(axis=0) + 1)
    corners = center + np.array([[-w, -h], [w, -h], [w, h], [-w, h]]) / 2
    return Polygon(rotate(corners, theta, center))


def _edge_class(a: np.ndarray, b: np.ndarray, snap_deg: float, oblique_min: float) -> str:
    d = b - a
    angle = np.degrees(np.arctan2(d[1], d[0])) % 180
    from_h, from_v = min(angle, 180 - angle), abs(angle - 90)
    if from_h <= snap_deg:
        return "H"
    if from_v <= snap_deg:
        return "V"
    if np.hypot(*d) < oblique_min:  # a raster step or a cut corner, not a real oblique wall
        return "H" if from_h < from_v else "V"
    return "O"


def _edge(cls: str, a: np.ndarray, b: np.ndarray) -> dict:
    offset = None if cls == "O" else (a[1] + b[1]) / 2 if cls == "H" else (a[0] + b[0]) / 2
    return {"cls": cls, "a": a, "b": b, "len": float(np.hypot(*(b - a))), "off": offset}


def _join(first: dict, second: dict) -> dict:
    """One H or V line from two consecutive edges of that class: the offsets weighted by length."""
    total = first["len"] + second["len"]
    offset = (first["off"] * first["len"] + second["off"] * second["len"]) / max(total, 1e-9)
    return {"cls": first["cls"], "a": first["a"], "b": second["b"], "len": total, "off": offset}


def _merge(edges: list[dict]) -> list[dict]:
    out = []
    for edge in edges:
        if out and out[-1]["cls"] == edge["cls"] != "O":
            out[-1] = _join(out[-1], edge)
        else:
            out.append(edge)
    if len(out) > 1 and out[0]["cls"] == out[-1]["cls"] != "O":
        out[0] = _join(out.pop(), out[0])
    return out


def _line(edge: dict) -> tuple[np.ndarray, np.ndarray]:
    """(point, unit direction) of the line an edge stands for."""
    if edge["cls"] == "H":
        return np.array([0.0, edge["off"]]), np.array([1.0, 0.0])
    if edge["cls"] == "V":
        return np.array([edge["off"], 0.0]), np.array([0.0, 1.0])
    d = edge["b"] - edge["a"]
    return edge["a"], d / max(np.hypot(*d), 1e-9)


def snap_rectilinear(points: np.ndarray, eps: float, params: RegularizeParams) -> np.ndarray | None:
    """Vertices of the snapped polygon from DP vertices in the θ-aligned frame (§7.2, steps 2-5); None when fewer than
    three edges are left."""
    edge_min = max(params.edge_min_px, 2 * eps)
    oblique_min = max(params.oblique_min_px, 4 * eps)
    pairs = [(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]
    edges = _merge([_edge(_edge_class(a, b, params.snap_deg, oblique_min), a, b) for a, b in pairs])
    while len(edges) > 4:
        shortest = int(np.argmin([e["len"] for e in edges]))
        if edges[shortest]["len"] >= edge_min:
            break
        del edges[shortest]
        edges = _merge(edges)
    if len(edges) < 3:
        return None
    out = []
    for i, edge in enumerate(edges):
        (p1, d1), (p2, d2) = _line(edges[i - 1]), _line(edge)
        system = np.array([d1, -d2]).T
        point = edge["a"]
        if abs(np.linalg.det(system)) >= 1e-6:
            crossing = p1 + np.linalg.solve(system, p2 - p1)[0] * d1
            if np.hypot(*(crossing - edge["a"])) <= 2 * eps + 2:  # a far crossing means nearly parallel lines
                point = crossing
        out.append(point)
    return np.array(out)


def _acceptable(polygon: Polygon | None, outline: Polygon) -> bool:
    return (
        polygon is not None
        and isinstance(polygon, Polygon)
        and polygon.is_valid
        and polygon.area > 0
        and iou(polygon, outline) >= MIN_IOU
    )


def select(candidates: list[tuple[str, Polygon | None]], outline: Polygon) -> Regularized:
    """The first acceptable candidate (§7.3), else the contour."""
    for path, polygon in candidates:
        if _acceptable(polygon, outline):
            return Regularized(polygon, path)
    return Regularized(outline, "contour")


def _polygon(points: np.ndarray | None) -> Polygon | None:
    if points is None or len(np.unique(points, axis=0)) < 3:
        return None
    return Polygon(points)


def regularize(
    mask: np.ndarray, theta: float, params: RegularizeParams | None = None, contour: np.ndarray | None = None
) -> Regularized:
    """Regularised polygon of one instance mask with orientation theta (radians, image coordinates), in the frame of
    the mask; contour: its marching-squares ring when already computed."""
    params = params or RegularizeParams()
    outline = outline_polygon(mask_to_exterior(mask) if contour is None else contour)
    rows, cols = np.nonzero(mask)
    box = moment_box(rows, cols, theta)
    if iou(box, outline) >= params.rect_iou:
        return Regularized(box, "box")
    eps = params.dp_k * np.sqrt(outline.area)
    center = np.asarray(outline.centroid.coords[0])
    simplified = largest_polygon(
        Polygon(rotate(vertices(outline), -theta, center)).simplify(eps, preserve_topology=True)
    )
    if simplified is None:
        return Regularized(outline, "contour")
    dp_points = vertices(simplified)
    snapped_points = snap_rectilinear(dp_points, eps, params)
    snapped = None if snapped_points is None else _polygon(rotate(snapped_points, theta, center))
    if snapped is not None:
        snapped = snapped.simplify(COLLINEAR_TOL, preserve_topology=True)
    return select([("snap", snapped), ("dp", _polygon(rotate(dp_points, theta, center)))], outline)
