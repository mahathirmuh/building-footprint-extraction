"""Ground truth per split and prediction files in COCO format (docs/EVALUATION.md §1, §2.2).

The ground truth of a split, outputs/gt/<dataset>/<split>[_<variant>].json, holds every image of the split (including
images without buildings) and the annotations pycocotools can evaluate:
- annotation ids are shifted by +1 when the source file starts them at 0, as CrowdAI val and the WHU conversion do:
  COCOeval stores the matched GT id in ``dtm`` and reads 0 as "not matched", so a detection matched with GT id 0 would
  count as a false positive. The offset is decided per source file, so all splits of one file share it;
- WHU negative samples (one annotation with an empty segmentation per patch with ≤ 5 building pixels) are dropped and
  their images kept: pycocotools cannot rasterise them, and the vendored metrics treat them as no GT anyway;
- every other annotation is kept as released, including CrowdAI's zero-area rings (decision of 30 Oct 2026).

Predictions are COCO results with one ring each, closed like the GT (both datasets repeat the first vertex), a
``score`` and a ``bbox``: pycocotools ``loadRes`` reads polygon results only when they carry a bbox, and takes the bbox
area as the detection area. check_predictions() enforces §1 strictly for our own outputs and leniently for released
ones (HiSup), where only rings that pycocotools cannot read are dropped and the rest is counted.
"""

from __future__ import annotations

import contextlib
import copy
import io
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from shapely.geometry import Polygon

from densepoly import paths
from densepoly.data.crowdai import polygon_conventions, valid_rings
from densepoly.utils.io import read_json, read_lines, write_json
from densepoly.utils.provenance import git_revision, timestamp

REQUIRED_FIELDS = ("image_id", "category_id", "segmentation", "score", "bbox")
OWN_FIELDS = ("theta_deg", "kappa")  # part of our prediction format, absent from released predictions
# Problems check_predictions() counts; in strict mode each is an error, in lenient mode only unreadable drops.
PROBLEMS = (
    "unreadable",
    "too_few_points",
    "multi_ring",
    "closing_vertex",
    "out_of_frame",
    "bbox_mismatch",
    "zero_area",
    "invalid",
)


def split_file_names(dataset: str, split: str, splits_dir: Path | None = None) -> list[str] | None:
    """File names of a split's images, in split-file order; None for val_full (the whole CrowdAI val)."""
    if split == "val_full":
        return None
    return read_lines(paths.split_list(dataset, split, splits_dir))


def is_negative_sample(ann: dict) -> bool:
    """WHU negative sample: an annotation whose segmentation has no ring."""
    segmentation = ann["segmentation"]
    return isinstance(segmentation, list) and not any(segmentation)


def pixel_area(segmentation: list, height: int, width: int) -> int:
    """Pixels pycocotools gives a polygon annotation; 0 for a zero-area ring (it cannot match anything)."""
    if isinstance(segmentation, dict):
        raise ValueError("RLE ground truth is not expected here")
    rings = valid_rings(segmentation)
    if not rings:
        return 0
    return int(mask_utils.area(mask_utils.merge(mask_utils.frPyObjects(rings, height, width))))


def id_offset(coco_data: dict) -> int:
    """+1 when the source file numbers annotations from 0 (see the module docstring), else 0."""
    ids = [a["id"] for a in coco_data["annotations"]]
    if not ids:
        return 0
    if min(ids) < 0 or len(set(ids)) != len(ids):
        raise ValueError("annotation ids must be unique and non-negative")
    return 1 if min(ids) == 0 else 0


def subset_gt(coco_data: dict, file_names: list[str] | None, offset: int) -> tuple[dict, dict]:
    """(ground truth of the images named, in that order, with annotation ids + offset and without negative samples;
    counts for its "info"). file_names None keeps every image."""
    images = coco_data["images"]
    if file_names is not None:
        by_name = {img["file_name"]: img for img in images}
        missing = [name for name in file_names if name not in by_name]
        if missing:
            raise ValueError(f"{len(missing)} split images are not in the ground truth, e.g. {missing[:3]}")
        images = [by_name[name] for name in file_names]
    sizes = {img["id"]: (img["height"], img["width"]) for img in images}
    anns = [a for a in coco_data["annotations"] if a["image_id"] in sizes]
    negative = sum(1 for a in anns if is_negative_sample(a))
    anns = [{**a, "id": a["id"] + offset} for a in anns if not is_negative_sample(a)]
    annotated = {a["image_id"] for a in anns}
    conventions = polygon_conventions(anns)
    closed, opened = conventions["closing_vertex_repeated"], conventions["closing_vertex_not_repeated"]
    counts = {
        "images": len(images),
        "images_without_annotations": sum(1 for img in images if img["id"] not in annotated),
        "annotations": len(anns),
        "dropped_negative_samples": negative,
        "ann_id_offset": offset,
        "multi_ring_annotations": sum(1 for a in anns if len(a["segmentation"]) > 1),
        "zero_area_rings": conventions["zero_area"],
        "empty_masks": sum(1 for a in anns if pixel_area(a["segmentation"], *sizes[a["image_id"]]) == 0),
        "area_field_zero": sum(1 for a in anns if a.get("area", 0) == 0),
        "closing_vertex_repeated": closed,
        "closing_vertex_not_repeated": opened,
        "ring_convention": "closed" if closed >= opened else "open",
    }
    subset = {"images": images, "annotations": anns, "categories": coco_data["categories"]}
    return subset, counts


def prepare_gt(
    dataset: str,
    split: str,
    variant: str | None = None,
    splits_dir: Path | None = None,
    coco_data: dict | None = None,
) -> dict:
    """Write paths.gt_file(dataset, split, variant) and return its "info". coco_data: the split's source annotation
    file (paths.annotation_file) when already loaded; the CrowdAI train file is ±0.7 GB."""
    source = paths.annotation_file(dataset, split, variant)
    coco_data = coco_data if coco_data is not None else read_json(source)
    subset, counts = subset_gt(coco_data, split_file_names(dataset, split, splits_dir), id_offset(coco_data))
    try:
        source_name = source.relative_to(paths.data_root()).as_posix()
    except ValueError:
        source_name = str(source)
    info = {
        "dataset": dataset,
        "split": split,
        "variant": variant,
        "source": source_name,
        "created": timestamp(),
        "git": git_revision(),
        **counts,
    }
    write_json(paths.gt_file(dataset, split, variant), {"info": info, **subset}, indent=None)
    return info


def load_gt(dataset: str, split: str, variant: str | None = None) -> dict:
    """The evaluated ground truth of a split, as written by prepare_gt()."""
    return read_json(paths.gt_file(dataset, split, variant))


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def coco_from_dict(data: dict) -> COCO:
    """pycocotools COCO over a deep copy of data: some metrics (polis.py) rewrite annotations in place."""
    coco = COCO()
    coco.dataset = copy.deepcopy(data)
    with _quiet():
        coco.createIndex()
    return coco


def load_results(coco_gt: COCO, results: list[dict]) -> COCO:
    """coco_gt.loadRes over a deep copy of results; no result gives an empty result set (loadRes fails on [])."""
    if results:
        with _quiet():
            return coco_gt.loadRes(copy.deepcopy(results))
    empty = COCO()
    empty.dataset = {
        "images": list(coco_gt.dataset["images"]),
        "categories": copy.deepcopy(coco_gt.dataset["categories"]),
        "annotations": [],
    }
    with _quiet():
        empty.createIndex()
    return empty


def ring_bbox(points: np.ndarray) -> list[float]:
    """COCO bbox [x, y, w, h] of (N, 2) points."""
    (x0, y0), (x1, y1) = points.min(axis=0), points.max(axis=0)
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def to_result(
    image_id: int,
    points,
    score: float,
    category_id: int,
    theta_deg: float | None = None,
    kappa: float | None = None,
    closed: bool = True,
    fallback: bool = False,
) -> dict:
    """Our prediction format (docs/EVALUATION.md §1) for one polygon: points (N ≥ 3, 2) as (x, y) in the original
    tile frame, without the closing vertex, which is appended when closed (the GT convention of both datasets).
    fallback: the polygon is a fallback of docs/METHOD.md §7.3, counted as n_fallback by the evaluation."""
    pts = np.asarray(points, float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        raise ValueError("a polygon needs at least 3 (x, y) points")
    ring = np.vstack([pts, pts[:1]]) if closed else pts
    return {
        "image_id": int(image_id),
        "category_id": int(category_id),
        "segmentation": [ring.ravel().tolist()],
        "score": float(score),
        "bbox": ring_bbox(pts),
        "theta_deg": None if theta_deg is None else float(theta_deg),
        "kappa": None if kappa is None else float(kappa),
        "fallback": bool(fallback),
    }


def distinct_points(ring) -> int:
    """Distinct vertices of a flat COCO ring (the repeated closing vertex is not counted twice)."""
    return len(np.unique(np.reshape(np.asarray(ring, float), (-1, 2)), axis=0))


def shapely_readable(results: list[dict]) -> list[dict]:
    """Results whose ring shapely 2 can build as a polygon (≥ 3 distinct points), for polis.py and angle_eval.py.

    The lenient check keeps rings such as [a, b, a] (6 numbers, 2 distinct points) because pycocotools reads them as
    empty masks, as in the evaluation of whoever released them; shapely 2 refuses them."""
    return [r for r in results if all(distinct_points(ring) >= 3 for ring in r["segmentation"])]


def _ring_problems(rings: list, bbox, width: int, height: int, closed: bool) -> list[str]:
    xy = np.asarray(rings[0], float).reshape(-1, 2)
    problems = ["multi_ring"] if len(rings) > 1 else []
    if any(len(ring) <= 4 for ring in rings):  # pycocotools reads 4 numbers as a bbox and fails on fewer
        return [*problems, "unreadable"]
    is_closed = len(xy) > 1 and np.array_equal(xy[0], xy[-1])
    pts = xy[:-1] if is_closed else xy
    if len(np.unique(pts, axis=0)) < 3:
        return [*problems, "too_few_points"]
    if is_closed != closed:
        problems.append("closing_vertex")
    eps = 1e-6
    if pts.min() < -eps or pts[:, 0].max() > width + eps or pts[:, 1].max() > height + eps:
        problems.append("out_of_frame")
    if len(bbox) != 4 or not np.allclose(bbox, ring_bbox(pts), rtol=0, atol=eps):
        problems.append("bbox_mismatch")
    polygon = Polygon(pts)
    if polygon.convex_hull.area == 0:  # all points on one line (a symmetric bow tie also has a shoelace area of 0)
        problems.append("zero_area")
    elif not polygon.is_valid:
        problems.append("invalid")
    return problems


def check_predictions(results: list[dict], gt: dict, strict: bool = True) -> tuple[list[dict], dict]:
    """(results kept, counts) after checking results against docs/EVALUATION.md §1 for the ground truth gt.

    A missing field, an unknown image or category, a non-finite score or a malformed segmentation is always an error.
    The PROBLEMS are errors in strict mode (our outputs). In lenient mode (released predictions) only a ring of at most
    4 numbers is dropped, since pycocotools reads 4 numbers as a bbox and crashes; every other problem is counted and
    those predictions are evaluated as released. A ring with < 3 distinct points but ≥ 6 numbers is such a case: an
    empty mask (a false positive) for pycocotools, left out of polis.py and angle_eval.py only (shapely_readable)."""
    images = {img["id"]: img for img in gt["images"]}
    categories = {c["id"] for c in gt["categories"]}
    closed = gt.get("info", {}).get("ring_convention", "closed") == "closed"
    required = REQUIRED_FIELDS + (OWN_FIELDS if strict else ())
    counts, examples, kept = Counter(), defaultdict(list), []
    for i, result in enumerate(results):
        missing = [key for key in required if key not in result]
        if missing:
            raise ValueError(f"result {i} lacks {missing}")
        if result["image_id"] not in images:
            raise ValueError(f"result {i}: image_id {result['image_id']} is not in the ground truth")
        if result["category_id"] not in categories:
            raise ValueError(f"result {i}: category_id {result['category_id']} is not in the ground truth")
        if not math.isfinite(result["score"]):
            raise ValueError(f"result {i}: score {result['score']} is not finite")
        rings = result["segmentation"]
        if not isinstance(rings, list) or not rings or any(not isinstance(r, list) or len(r) % 2 for r in rings):
            raise ValueError(f"result {i}: segmentation must be a list of flat [x1, y1, x2, y2, ...] rings")
        image = images[result["image_id"]]
        problems = _ring_problems(rings, result["bbox"], image["width"], image["height"], closed)
        for problem in problems:
            counts[problem] += 1
            if len(examples[problem]) < 3:
                examples[problem].append(i)
        if not strict and "unreadable" in problems:
            continue
        kept.append(result)
    if strict and counts:
        found = "; ".join(f"{p}: {counts[p]} (e.g. results {examples[p]})" for p in PROBLEMS if counts[p])
        raise ValueError(f"predictions break docs/EVALUATION.md §1: {found}")
    report = {"results": len(results), "kept": len(kept), "dropped": len(results) - len(kept)}
    return kept, {**report, **{p: counts[p] for p in PROBLEMS}}


def restrict_to_images(results: list[dict], image_ids) -> list[dict]:
    """Results of some images only, e.g. HiSup's val predictions restricted to val_clean."""
    keep = set(image_ids)
    return [r for r in results if r["image_id"] in keep]


def gt_as_predictions(gt: dict) -> list[dict]:
    """Every GT annotation as a result with score 1, its first ring and that ring's bbox (docs/EVALUATION.md §6.1).

    Results have one ring, as in our prediction format; the vendored cIoU.py cannot read more. In both datasets the
    other rings of a GT annotation are holes inside the first one, and pycocotools merges rings by union, so the mask
    and every metric stay the same (checked on the WHU test GT, 30 Oct 2026)."""
    results = []
    for ann in gt["annotations"]:
        ring = ann["segmentation"][0]
        points = np.reshape(ring, (-1, 2)).astype(float)
        results.append(
            {
                "image_id": ann["image_id"],
                "category_id": ann["category_id"],
                "segmentation": [list(ring)],
                "score": 1.0,
                "bbox": ring_bbox(points),
                "theta_deg": None,
                "kappa": None,
            }
        )
    return results


def empty_mask_ids(gt: dict) -> list[int]:
    """Ids of the annotations whose pycocotools mask covers no pixel (zero-area rings): nothing can match them."""
    sizes = {img["id"]: (img["height"], img["width"]) for img in gt["images"]}
    return [a["id"] for a in gt["annotations"] if pixel_area(a["segmentation"], *sizes[a["image_id"]]) == 0]


def without_annotations(gt: dict, ann_ids) -> dict:
    """The ground truth without some annotations (images stay), e.g. without the empty masks for §6.1."""
    drop = set(ann_ids)
    return {**gt, "annotations": [a for a in gt["annotations"] if a["id"] not in drop]}
