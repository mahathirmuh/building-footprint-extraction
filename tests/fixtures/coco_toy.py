"""Tiny COCO scenes for the evaluation tests: polygons on 64 × 64 images, closed rings as in both datasets."""

from __future__ import annotations

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon

from densepoly.eval import coco_io

SIZE = 64
CATEGORY = 100
RECT = [(8, 20), (48, 20), (48, 36), (8, 36)]


def square(x, y, side):
    return [(x, y), (x + side, y), (x + side, y + side), (x, y + side)]


def rotated(points, degrees):
    return list(affinity.rotate(Polygon(points), degrees, origin="centroid").exterior.coords)[:-1]


def closed_ring(points):
    return [float(v) for point in [*points, points[0]] for v in point]


def make_gt(scenes: dict[int, list]) -> dict:
    """Ground truth with one image per key and one annotation per point list (ids from 1)."""
    images = [{"id": i, "file_name": f"{i}.png", "height": SIZE, "width": SIZE} for i in scenes]
    annotations = []
    for image_id, polygons in scenes.items():
        for points in polygons:
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": CATEGORY,
                    "segmentation": [closed_ring(points)],
                    "area": float(Polygon(points).area),
                    "bbox": coco_io.ring_bbox(np.asarray(points, float)),
                    "iscrowd": 0,
                }
            )
    categories = [{"id": CATEGORY, "name": "building"}]
    return {
        "info": {"ring_convention": "closed"},
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def make_results(scenes: dict[int, list], score: float = 0.9, theta_deg: float | None = None) -> list[dict]:
    """Our prediction format, one result per point list; scores decrease slightly in list order."""
    results = []
    for image_id, polygons in scenes.items():
        for points in polygons:
            results.append(coco_io.to_result(image_id, points, score - 0.01 * len(results), CATEGORY, theta_deg))
    return results
