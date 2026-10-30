"""Pixel metrics of the union masks of each image (docs/EVALUATION.md §2.1, §2.2): IoU, F1, Boundary IoU and the main
C-IoU, computed on every image of a split.

The masks are the pycocotools rasters of all polygon rings of an image, merged (holes filled), as in the vendored
cIoU.py. C-IoU uses its formula, calc_IoU × (1 − |N − N_GT| / (N + N_GT)) with N the vertices of the first ring of
every polygon (closing vertex included), on all images: an image without prediction scores 0 when it has GT and 1
when both are empty, and an image with predictions but no GT scores 0. On the images that have predictions the
values equal those of the vendored cIoU.py, which gives CIoU_p2p. Boundary IoU (Cheng et al. 2021) is the IoU of the
inner bands, 2% of the image diagonal wide, of the two masks (boundary-iou-api mask_to_boundary); 1 when both bands
are empty.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from boundary_iou.utils.boundary_utils import mask_to_boundary
from pix2poly_metrics.cIoU import calc_IoU
from pycocotools import mask as mask_utils

from densepoly.data.crowdai import valid_rings

DILATION_RATIO = 0.02


def union_mask(anns: list[dict], height: int, width: int) -> np.ndarray:
    """(H, W) bool union of all polygon rings of anns."""
    rings = [ring for ann in anns for ring in valid_rings(ann["segmentation"])]
    if not rings:
        return np.zeros((height, width), bool)
    return mask_utils.decode(mask_utils.merge(mask_utils.frPyObjects(rings, height, width))).astype(bool)


def first_ring_vertices(anns: list[dict]) -> int:
    """N of C-IoU: points of the first ring of every polygon, the closing vertex included (vendored cIoU.py)."""
    return sum(len(ann["segmentation"][0]) // 2 for ann in anns if ann["segmentation"])


def boundary_iou(gt: np.ndarray, pred: np.ndarray, dilation_ratio: float = DILATION_RATIO) -> float:
    """IoU of the inner boundary bands of two (H, W) masks; 1 when both bands are empty."""
    gt_band = mask_to_boundary(gt.astype(np.uint8), dilation_ratio) > 0
    pred_band = mask_to_boundary(pred.astype(np.uint8), dilation_ratio) > 0
    union = int(np.logical_or(gt_band, pred_band).sum())
    return 1.0 if union == 0 else int(np.logical_and(gt_band, pred_band).sum()) / union


def image_pixel_metrics(image: dict, gt_anns: list[dict], pred_anns: list[dict]) -> dict:
    """Pixel counts and per-image metrics of one image."""
    height, width = image["height"], image["width"]
    gt, pred = union_mask(gt_anns, height, width), union_mask(pred_anns, height, width)
    n, n_gt = first_ring_vertices(pred_anns), first_ring_vertices(gt_anns)
    iou = float(calc_IoU(pred, gt))
    return {
        "image_id": image["id"],
        "n_pred": len(pred_anns),
        "n_gt": len(gt_anns),
        "gt_px": int(gt.sum()),
        "pred_px": int(pred.sum()),
        "inter_px": int(np.logical_and(gt, pred).sum()),
        "iou": iou,
        "ciou": iou * (1 - abs(n - n_gt) / (n + n_gt + 1e-9)),
        "n_vertices": n,
        "n_vertices_gt": n_gt,
        "biou": boundary_iou(gt, pred),
    }


def pixel_metrics(gt: dict, results: list[dict]) -> list[dict]:
    """image_pixel_metrics of every image of gt, in gt order."""
    gt_by_image, pred_by_image = defaultdict(list), defaultdict(list)
    for ann in gt["annotations"]:
        gt_by_image[ann["image_id"]].append(ann)
    for result in results:
        pred_by_image[result["image_id"]].append(result)
    return [image_pixel_metrics(img, gt_by_image[img["id"]], pred_by_image[img["id"]]) for img in gt["images"]]


def _mean(values) -> float | None:
    return float(np.mean(values)) if len(values) else None


def summarize_pixel(rows: list[dict]) -> dict:
    """Split-level pixel metrics of per-image rows (all images, or those of one stratum)."""
    inter = sum(r["inter_px"] for r in rows)
    gt_px, pred_px = sum(r["gt_px"] for r in rows), sum(r["pred_px"] for r in rows)
    union = gt_px + pred_px - inter
    ratios = [r["n_vertices"] / r["n_vertices_gt"] for r in rows if r["n_vertices_gt"] > 0]
    return {
        "IoU": (1.0 if union == 0 else inter / union) if rows else None,
        "IoU_per_image": _mean([r["iou"] for r in rows]),
        "F1": (1.0 if gt_px + pred_px == 0 else 2 * inter / (gt_px + pred_px)) if rows else None,
        "BoundaryIoU": _mean([r["biou"] for r in rows]),
        "CIoU": _mean([r["ciou"] for r in rows]),
        "N_ratio": _mean(ratios),
        "n_images": len(rows),
        "n_images_with_pred": sum(1 for r in rows if r["n_pred"]),
    }
