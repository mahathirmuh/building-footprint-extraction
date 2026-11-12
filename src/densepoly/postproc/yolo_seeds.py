"""FULL+Y (docs/METHOD.md §8): oriented boxes of a detector as extra watershed seeds.

Two halves, neither of which needs the detector itself (Ultralytics is AGPL and stays in scripts/):
- labels: one oriented box per building, from the instance map alone (cv2.minAreaRect), in the Ultralytics OBB format;
- seeds: the centres of predicted boxes merged into the markers of §6.2 step 3, after the m_min filter, so that a
  13-pixel seed is not thrown away by a filter whose grid goes up to 16 pixels.
"""

from __future__ import annotations

import cv2
import numpy as np

from densepoly.postproc.markers import label4

SEED_RADIUS = 2  # a disk of 13 pixels


def instance_boxes(instance_map: np.ndarray, min_area: int = 1) -> list[np.ndarray]:
    """(4, 2) corners (x, y) of the minimum-area rectangle of every instance of at least min_area pixels, in the order of
    the instance ids. The rectangle encloses the pixels as squares (their corners), not just their centres."""
    boxes = []
    for instance in np.unique(instance_map):
        if instance == 0:
            continue
        rows, cols = np.nonzero(instance_map == instance)
        if rows.size < min_area:
            continue
        corners = np.concatenate([np.stack([cols + dx, rows + dy], axis=1) for dx in (0, 1) for dy in (0, 1)])
        boxes.append(cv2.boxPoints(cv2.minAreaRect(corners.astype(np.float32))).astype(np.float64))
    return boxes


def obb_label_lines(instance_map: np.ndarray, min_area: int = 1) -> list[str]:
    """The Ultralytics OBB label of a tile: "0 x1 y1 x2 y2 x3 y3 x4 y4" per building, coordinates divided by the tile
    width and height and clipped to [0, 1]."""
    height, width = instance_map.shape
    lines = []
    for box in instance_boxes(instance_map, min_area):
        normalized = np.clip(box / (width, height), 0.0, 1.0)
        lines.append("0 " + " ".join(f"{value:.6f}" for value in normalized.ravel()))
    return lines


def seed_centres(boxes: list, confidence: float) -> np.ndarray:
    """(n, 2) centres (x, y) of the predicted boxes [x, y, confidence, ...] whose confidence is at least `confidence`."""
    kept = [(x, y) for x, y, score, *_ in boxes if score >= confidence]
    return np.asarray(kept, float).reshape(-1, 2)


def _disk(shape: tuple[int, int], x: float, y: float, radius: int) -> np.ndarray:
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    return (rows - int(y)) ** 2 + (cols - int(x)) ** 2 <= radius**2


def merge_seeds(marker_labels: np.ndarray, fg: np.ndarray, centres: np.ndarray, radius: int = SEED_RADIUS) -> np.ndarray:
    """Markers with the detector's seeds merged in (§8), labelled 1..n:

    1. a centre inside the foreground becomes a seed, a disk cut to the foreground; other centres are ignored;
    2. a marker that holds two or more seeds is replaced by them (a merged house is split);
    3. a marker with one seed or none is kept as it is;
    4. a seed in a foreground component without any marker is added;
    a seed outside every marker, in a component that has one, changes nothing. A seed never paints over a marker that
    is kept or over an earlier seed.
    """
    height, width = fg.shape
    inside = [(x, y) for x, y in np.asarray(centres, float).reshape(-1, 2) if 0 <= x < width and 0 <= y < height]
    seeds = [(x, y) for x, y in inside if fg[int(y), int(x)]]
    if not seeds:
        return marker_labels
    components = label4(fg)
    with_marker = set(np.unique(components[marker_labels > 0]).tolist())
    in_marker: dict[int, list] = {}
    added = []
    for x, y in seeds:
        marker = int(marker_labels[int(y), int(x)])
        if marker:
            in_marker.setdefault(marker, []).append((x, y))
        elif int(components[int(y), int(x)]) not in with_marker:
            added.append((x, y))
    merged = marker_labels.copy()
    next_label = int(merged.max()) + 1
    for marker, held in in_marker.items():
        if len(held) >= 2:
            merged[merged == marker] = 0
            added += held
    for x, y in added:
        free = _disk(fg.shape, x, y, radius) & fg & (merged == 0)
        if free.any():
            merged[free] = next_label
            next_label += 1
    present = np.unique(merged)
    renumber = np.zeros(int(present.max()) + 1, np.int32)
    renumber[present[present > 0]] = np.arange(1, int((present > 0).sum()) + 1, dtype=np.int32)
    return renumber[merged]
