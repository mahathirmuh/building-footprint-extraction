"""Instance metrics from a one-to-one matching (docs/EVALUATION.md §2.1, §2.4).

Matching follows COCO: predictions in decreasing score each take the still unmatched GT of highest mask IoU, if that
IoU is ≥ 0.5 (exact ties go to the first GT). Unlike COCOeval there is no cap of 100 detections per image. Each pair
gives PoLiS (vendored compare_polys on the first rings), the max tangent angle error of the pair alone (vendored
compute_contour_metrics, so walls between touching houses are not merged away), the θ error (eval/orientation.py)
and the vertex matches of Vertex F1 (eval/vertex.py). Each GT gives recall (it has a pair), merge (a prediction
covers it, |g ∩ p| / |g| ≥ 0.5, and covers another GT too) and split (≥ 2 predictions lie mostly inside it,
|g ∩ p| / |p| ≥ 0.5). A GT with an empty mask only counts against recall; a prediction with an empty mask (possible
after the lenient check only) takes part in no merge or split.
"""

from __future__ import annotations

import math
import multiprocessing
from collections import defaultdict

import numpy as np
from pix2poly_metrics.angle_eval import compute_contour_metrics
from pix2poly_metrics.polis import compare_polys
from pycocotools import mask as mask_utils
from shapely.geometry import Polygon

from densepoly.data.crowdai import valid_rings
from densepoly.eval.orientation import polygon_theta_deg, theta_error_deg
from densepoly.eval.vertex import ring_vertices, vertex_matches

IOU_THRESHOLD = 0.5
COVER_THRESHOLD = 0.5


def instance_rle(segmentation: list, height: int, width: int) -> dict:
    """pycocotools RLE of one polygon annotation (its rings merged); an empty mask without a readable ring."""
    rings = valid_rings(segmentation)
    if not rings:
        return mask_utils.encode(np.zeros((height, width, 1), np.uint8, order="F"))[0]
    return mask_utils.merge(mask_utils.frPyObjects(rings, height, width))


def match(ious: np.ndarray, scores) -> list[tuple[int, int, float]]:
    """(prediction, GT, IoU) of the greedy one-to-one matching; ious is (n_pred, n_gt), indices are positions."""
    taken = np.zeros(ious.shape[1], bool)
    pairs = []
    for d in np.argsort(-np.asarray(scores, float), kind="mergesort"):
        if taken.all():
            break
        candidates = np.where(taken, -1.0, ious[d])
        g = int(np.argmax(candidates))
        if candidates[g] >= IOU_THRESHOLD:
            taken[g] = True
            pairs.append((int(d), g, float(candidates[g])))
    return pairs


def pair_measures(gt_ann: dict, result: dict) -> dict:
    """Geometric measures of one matched pair."""
    gt_ring, pred_ring = gt_ann["segmentation"][0], result["segmentation"][0]
    polis = compare_polys(Polygon(np.reshape(gt_ring, (-1, 2))), Polygon(np.reshape(pred_ring, (-1, 2))))
    angles = compute_contour_metrics(([gt_ann], [{"segmentation": result["segmentation"]}]))
    angles = [a for a in angles if a is not None]
    theta = result.get("theta_deg")
    gt_vertices, pred_vertices = ring_vertices(gt_ring), ring_vertices(pred_ring)
    return {
        "polis": float(polis),
        "mta": math.degrees(max(angles)) if angles else math.nan,
        "theta_err": math.nan if theta is None else theta_error_deg(theta, polygon_theta_deg(gt_ring)),
        "vertices_gt": len(gt_vertices),
        "vertices_pred": len(pred_vertices),
        "vertex_tp": vertex_matches(gt_vertices, pred_vertices),
    }


def evaluate_image(job: tuple[dict, list[dict], list[tuple[int, dict]]]) -> tuple[list[dict], list[dict]]:
    """(rows per GT, rows per pair) of one image; job = (image, GT annotations, [(result index, result), ...])."""
    image, gt_anns, preds = job
    height, width = image["height"], image["width"]
    gt_rles = [instance_rle(a["segmentation"], height, width) for a in gt_anns]
    pred_rles = [instance_rle(r["segmentation"], height, width) for _, r in preds]
    gt_px = np.asarray(mask_utils.area(gt_rles) if gt_rles else [], float)
    pred_px = np.asarray(mask_utils.area(pred_rles) if pred_rles else [], float)
    shape = (len(pred_rles), len(gt_rles))
    ious = np.asarray(mask_utils.iou(pred_rles, gt_rles, [0] * len(gt_rles)), float).reshape(shape)

    inter = np.zeros(shape)
    for d, g in zip(*np.nonzero(ious > 0), strict=True):
        inter[d, g] = mask_utils.area(mask_utils.merge([pred_rles[d], gt_rles[g]], intersect=True))
    with np.errstate(divide="ignore", invalid="ignore"):
        covers = (inter / gt_px[None, :] >= COVER_THRESHOLD) & (gt_px[None, :] > 0) & (pred_px[:, None] > 0)
        inside = (inter / pred_px[:, None] >= COVER_THRESHOLD) & (pred_px[:, None] > 0) & (gt_px[None, :] > 0)
    merging = covers.sum(axis=1) >= 2  # predictions that cover two or more GT
    merged = (covers & merging[:, None]).any(axis=0)
    split = inside.sum(axis=0) >= 2

    pairs = match(ious, [r["score"] for _, r in preds])
    matched = np.zeros(len(gt_anns), bool)
    pair_rows = []
    for d, g, iou in pairs:
        matched[g] = True
        index, result = preds[d]
        pair_rows.append(
            {
                "image_id": image["id"],
                "gt_id": gt_anns[g]["id"],
                "pred_index": index,
                "score": float(result["score"]),
                "iou": iou,
                **pair_measures(gt_anns[g], result),
            }
        )
    gt_rows = [
        {
            "image_id": image["id"],
            "gt_id": ann["id"],
            "gt_px": int(gt_px[g]),
            "matched": bool(matched[g]),
            "merged": bool(merged[g]),
            "split": bool(split[g]),
        }
        for g, ann in enumerate(gt_anns)
    ]
    return gt_rows, pair_rows


def instance_metrics(gt: dict, results: list[dict], workers: int = 1) -> tuple[list[dict], list[dict]]:
    """(rows per GT, rows per pair) of a split; pair rows refer to results by their index in results."""
    gt_by_image, pred_by_image = defaultdict(list), defaultdict(list)
    for ann in gt["annotations"]:
        gt_by_image[ann["image_id"]].append(ann)
    for index, result in enumerate(results):
        pred_by_image[result["image_id"]].append((index, result))
    jobs = [(img, gt_by_image[img["id"]], pred_by_image[img["id"]]) for img in gt["images"]]
    if workers > 1 and len(jobs) > 1:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            outputs = pool.map(evaluate_image, jobs, chunksize=max(1, len(jobs) // (8 * workers)))
    else:
        outputs = [evaluate_image(job) for job in jobs]
    return [row for rows, _ in outputs for row in rows], [row for _, rows in outputs for row in rows]


def _mean(values) -> float | None:
    return float(np.mean(values)) if len(values) else None


def summarize_instance(gt_rows: list[dict], pair_rows: list[dict]) -> dict:
    """Split-level instance metrics of the rows of all images, or of the images of one stratum."""
    n_gt = len(gt_rows)
    mta = [p["mta"] for p in pair_rows if not math.isnan(p["mta"])]
    theta = [p["theta_err"] for p in pair_rows if not math.isnan(p["theta_err"])]
    vertices = sum(p["vertices_gt"] + p["vertices_pred"] for p in pair_rows)
    return {
        "recall_inst50": sum(r["matched"] for r in gt_rows) / n_gt if n_gt else None,
        "merge_rate": sum(r["merged"] for r in gt_rows) / n_gt if n_gt else None,
        "split_rate": sum(r["split"] for r in gt_rows) / n_gt if n_gt else None,
        "PoLiS_paired": _mean([p["polis"] for p in pair_rows]),
        "MTA_inst": _mean(mta),
        "vertex_F1": 2 * sum(p["vertex_tp"] for p in pair_rows) / vertices if vertices else None,
        "theta_err_median": float(np.median(theta)) if theta else None,
        "theta_err_p90": float(np.percentile(theta, 90)) if theta else None,
        "n_gt": n_gt,
        "n_pairs": len(pair_rows),
        "n_pairs_without_mta": len(pair_rows) - len(mta),
    }
