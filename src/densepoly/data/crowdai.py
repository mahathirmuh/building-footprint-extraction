"""CrowdAI Mapping Challenge: download check, annotation statistics and instance rasterisation (docs/DATA.md §2).

Instance maps are painted by decreasing pixel area, so a small building drawn over a big one stays visible
(docs/DATA.md §2.3); ``tests/fixtures/synthetic.py`` follows the same rule, and the tests check both against each other.
The COCO helpers here do not depend on the dataset and are reused for the WHU protocol A ground truth.
"""

from __future__ import annotations

import contextlib
import io
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def load_coco(path: Path) -> COCO:
    """Load a COCO annotation file without pycocotools' progress prints."""
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(path))


def list_images(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def valid_rings(segmentation: list) -> list[list[float]]:
    """Polygon rings with at least 3 points; pycocotools would read a 4-number ring as a bounding box."""
    return [ring for ring in segmentation if len(ring) >= 6 and len(ring) % 2 == 0]


def annotation_mask(segmentation, height: int, width: int) -> np.ndarray | None:
    """Binary mask of one annotation (polygon rings are merged; RLE is decoded); None without a usable ring."""
    if isinstance(segmentation, dict):
        rle = segmentation
        if isinstance(rle["counts"], list):
            rle = mask_utils.frPyObjects(rle, height, width)
        return mask_utils.decode(rle).astype(bool)
    rings = valid_rings(segmentation)
    if not rings:
        return None
    return mask_utils.decode(mask_utils.merge(mask_utils.frPyObjects(rings, height, width))).astype(bool)


@dataclass
class RasterResult:
    instance_map: np.ndarray  # (H, W) uint16, value i + 1 = annotation_ids[i]
    annotation_ids: list[int]
    n_skipped: int  # annotations without a ring of >= 3 points
    n_empty: int  # usable polygon that covers no pixel centre
    overlap_px: int  # extra claims on pixels covered by several annotations
    n_hidden: int  # annotations entirely painted over by smaller ones


def rasterize_instances(anns: list[dict], height: int, width: int) -> RasterResult:
    """Instance map of one image: kept annotations sorted by id get values 1..n, painted by decreasing pixel area."""
    masks, ids, skipped, empty = [], [], 0, 0
    for ann in sorted(anns, key=lambda a: a["id"]):
        mask = annotation_mask(ann["segmentation"], height, width)
        if mask is None:
            skipped += 1
        elif not mask.any():
            empty += 1
        else:
            masks.append(mask)
            ids.append(ann["id"])
    if len(masks) > np.iinfo(np.uint16).max:
        raise ValueError("too many instances for a uint16 map")
    out = np.zeros((height, width), np.uint16)
    for i in sorted(range(len(masks)), key=lambda i: (-int(masks[i].sum()), i)):
        out[masks[i]] = i + 1
    claimed = sum(int(m.sum()) for m in masks)
    visible = np.bincount(out.ravel(), minlength=len(masks) + 1)[1:]
    return RasterResult(out, ids, skipped, empty, claimed - int((out > 0).sum()), int((visible == 0).sum()))


def polygon_conventions(anns: list[dict]) -> dict:
    """How the GT rings are written: repeated closing vertex or not, winding, and vertex counts."""
    closed = opened = clockwise = counter = degenerate = 0
    vertices = []
    for ann in anns:
        if isinstance(ann["segmentation"], dict):
            continue
        for ring in valid_rings(ann["segmentation"]):
            xy = np.asarray(ring, float).reshape(-1, 2)
            is_closed = len(xy) > 3 and np.array_equal(xy[0], xy[-1])
            closed += is_closed
            opened += not is_closed
            pts = xy[:-1] if is_closed else xy
            vertices.append(len(pts))
            twice_area = np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
            # y points down in image coordinates, so a positive shoelace sum means clockwise on screen
            clockwise += twice_area > 0
            counter += twice_area < 0
            degenerate += twice_area == 0
    counts = np.asarray(vertices)
    return {
        "rings": len(vertices),
        "closing_vertex_repeated": int(closed),
        "closing_vertex_not_repeated": int(opened),
        "clockwise_on_screen": int(clockwise),
        "counterclockwise_on_screen": int(counter),
        "zero_area": int(degenerate),
        "vertices_mean": float(counts.mean()) if counts.size else None,
        "vertices_median": float(np.median(counts)) if counts.size else None,
        "vertices_max": int(counts.max()) if counts.size else None,
    }


def annotation_stats(coco: COCO) -> dict:
    """Counts that docs/DATA.md §2.3 asks to record for one annotation file."""
    images, anns = coco.dataset["images"], coco.dataset["annotations"]
    per_image = Counter(a["image_id"] for a in anns)
    polygons = [a for a in anns if not isinstance(a["segmentation"], dict)]
    counts = np.array([per_image.get(img["id"], 0) for img in images])
    return {
        "images": len(images),
        "annotations": len(anns),
        "images_without_annotations": int((counts == 0).sum()),
        "annotations_per_image": {"mean": float(counts.mean()), "max": int(counts.max())} if counts.size else None,
        "categories": coco.dataset.get("categories", []),
        "rle_annotations": len(anns) - len(polygons),
        "iscrowd_annotations": sum(1 for a in anns if a.get("iscrowd", 0)),
        "annotations_without_valid_ring": sum(1 for a in polygons if not valid_rings(a["segmentation"])),
        "multi_ring_annotations": sum(1 for a in polygons if len(valid_rings(a["segmentation"])) > 1),
        "polygons": polygon_conventions(anns),
    }


def check_files(coco: COCO, image_dir: Path) -> dict:
    """Image files listed in the annotation file versus files present on disk."""
    listed = {img["file_name"] for img in coco.dataset["images"]}
    on_disk = set(list_images(image_dir)) if image_dir.is_dir() else set()
    missing, extra = sorted(listed - on_disk), sorted(on_disk - listed)
    return {
        "files_on_disk": len(on_disk),
        "missing_files": len(missing),
        "extra_files": len(extra),
        "missing_examples": missing[:10],
        "extra_examples": extra[:10],
    }
