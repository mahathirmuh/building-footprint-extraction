"""Statistics of the evaluation (docs/EVALUATION.md §4).

COCO AP over any list of images, duplicates allowed, from the evalImgs of a single COCOeval.evaluate(). evaluate()
drops duplicate imgIds (np.unique), so a bootstrap resample cannot simply be passed as params.imgIds. Instead the
cached evalImgs are re-indexed for the image list and pycocotools' own COCOeval.accumulate() and summarize() run on
them unchanged: with the identity list they give stats exactly. The evalImgs of the vendored boundary-iou-api COCOeval
have the same structure, so AP_boundary goes the same way (its own accumulate() fails on numpy ≥ 1.24).

Cluster bootstrap by image: a replicate draws the images of a set with replacement and is stored as the number of
times each image was drawn. One weight matrix serves every method, seed and metric, so comparisons are paired. A
ratio metric of a replicate is Σ weight · numerator / Σ weight · denominator over per-image terms taken from the
saved tables of run_eval (paired metrics: summed per image over the pairs whose GT both methods matched); AP runs
coco_ap on the resampled image list. Δ = metric(B) − metric(A) is taken per seed and averaged over seeds; the CI is
the 2.5–97.5 percentile range of that average over the replicates.
"""

from __future__ import annotations

import contextlib
import copy
import io
import math
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pycocotools.cocoeval import COCOeval

from densepoly import paths

STAT_NAMES = ("AP", "AP50", "AP75", "AP_S", "AP_M", "AP_L", "AR1", "AR10", "AR100", "AR_S", "AR_M", "AR_L")


@dataclass
class EvalCache:
    """What COCOeval.accumulate() needs from one evaluate(): evalImgs, ordered [category][area][image], and the
    parameters of that evaluation (img_ids sorted and unique, as evaluate() leaves them)."""

    eval_imgs: list
    params: object

    @property
    def img_ids(self) -> list[int]:
        return [int(i) for i in self.params.imgIds]

    def positions(self, image_ids) -> list[int]:
        """Positions of image ids in img_ids, in the order given (duplicates kept)."""
        index = {img_id: i for i, img_id in enumerate(self.img_ids)}
        return [index[int(i)] for i in image_ids]


def eval_cache(evaluator) -> EvalCache:
    """EvalCache of a COCOeval (pycocotools or vendored boundary-iou-api) after evaluate()."""
    return EvalCache(evaluator.evalImgs, copy.deepcopy(evaluator._paramsEval))


def _reindexed(cache: EvalCache, positions, area_labels=None, max_dets=None) -> COCOeval:
    """A pycocotools COCOeval holding the evalImgs of the images at positions, ready for accumulate()."""
    p = cache.params
    n_cats = len(p.catIds) if p.useCats else 1
    n_areas, n_images = len(p.areaRng), len(p.imgIds)
    areas = range(n_areas) if area_labels is None else [p.areaRngLbl.index(label) for label in area_labels]
    positions = list(positions)
    eval_imgs = [
        cache.eval_imgs[k * n_areas * n_images + a * n_images + i]
        for k in range(n_cats)
        for a in areas
        for i in positions
    ]
    params = copy.copy(p)
    params.imgIds = list(range(len(positions)))
    params.areaRng = [p.areaRng[a] for a in areas]
    params.areaRngLbl = [p.areaRngLbl[a] for a in areas]
    if max_dets is not None:
        params.maxDets = list(max_dets)
    params.iouType = "segm"  # summarize() only tells keypoints from boxes and masks
    shell = COCOeval.__new__(COCOeval)
    shell.evalImgs, shell.params, shell._paramsEval, shell.eval, shell.stats = eval_imgs, params, params, {}, []
    return shell


def coco_stats(cache: EvalCache, positions=None) -> dict[str, float]:
    """The 12 COCO summary numbers (STAT_NAMES) over the images at positions; None means all images."""
    shell = _reindexed(cache, range(len(cache.params.imgIds)) if positions is None else positions)
    with contextlib.redirect_stdout(io.StringIO()):
        shell.accumulate()
        shell.summarize()
    return dict(zip(STAT_NAMES, (float(s) for s in shell.stats), strict=True))


def coco_ap(cache: EvalCache, positions=None) -> float:
    """AP alone (IoU 0.50:0.95, all areas, 100 detections), equal to coco_stats()["AP"] but accumulating that one
    setting only, for the bootstrap."""
    shell = _reindexed(cache, range(len(cache.params.imgIds)) if positions is None else positions, ["all"], [100])
    with contextlib.redirect_stdout(io.StringIO()):
        shell.accumulate()
    precision = shell.eval["precision"]
    valid = precision[precision > -1]
    return float(np.mean(valid)) if valid.size else -1.0


# Ratio metrics: per-image numerator and denominator terms (docs/EVALUATION.md §2.1, §3), from run_eval's tables.
RATIO_METRICS = (
    "IoU",
    "IoU_per_image",
    "F1",
    "BoundaryIoU",
    "CIoU",
    "N_ratio",
    "CIoU_p2p",
    "PoLiS",
    "MTA",
    "recall_inst50",
    "merge_rate",
    "split_rate",
    "PoLiS_paired",
    "MTA_inst",
    "vertex_F1",
)
PAIRED_METRICS = ("PoLiS_paired", "MTA_inst", "vertex_F1")  # on the GT matched by both methods (§2.4)
AP_METRICS = {"AP": "segm", "AP_boundary": "boundary"}
EMPTY_VALUES = {"IoU": 1.0, "F1": 1.0}  # value when every mask is empty (as in eval/pixel.py)


@dataclass
class RunTables:
    """The saved evaluation of one run on one split: run_eval's per-image table (indexed by image_id), its pairs and,
    when cached, the evalImgs of AP and AP_boundary."""

    per_image: pd.DataFrame
    pairs: pd.DataFrame
    caches: dict[str, EvalCache] | None = None


def load_run(dataset: str, exp: str, run: str, split: str, variant: str | None = None) -> RunTables:
    """RunTables of outputs/metrics/<dataset>/<exp>/<run>/<split>[_<variant>].*."""
    per_image = pd.read_parquet(paths.metrics_artifact(dataset, exp, run, split, "per_image.parquet", variant))
    pairs = pd.read_parquet(paths.metrics_artifact(dataset, exp, run, split, "pairs.parquet", variant))
    caches = None
    cache_file = paths.metrics_artifact(dataset, exp, run, split, "evalimgs.pkl", variant)
    if cache_file.exists():
        with cache_file.open("rb") as f:  # written by run_eval for this project only
            caches = pickle.load(f)
    return RunTables(per_image.set_index("image_id", drop=False), pairs, caches)


def common_gt(a: RunTables, b: RunTables, metric: str) -> set[int]:
    """GT ids matched by both runs; for MTA_inst also with a value in both (§2.4)."""
    pa, pb = a.pairs, b.pairs
    if metric == "MTA_inst":
        pa, pb = pa[pa["mta"].notna()], pb[pb["mta"].notna()]
    return set(pa["gt_id"]) & set(pb["gt_id"])


def _paired_terms(run: RunTables, metric: str, image_ids: list[int], gt_ids: set[int] | None):
    pairs = run.pairs if gt_ids is None else run.pairs[run.pairs["gt_id"].isin(gt_ids)]
    if metric == "MTA_inst":
        pairs = pairs[pairs["mta"].notna()]
    grouped = pairs.groupby("image_id")
    if metric == "PoLiS_paired":
        num, den = grouped["polis"].sum(), grouped.size()
    elif metric == "MTA_inst":
        num, den = grouped["mta"].sum(), grouped.size()
    else:  # vertex_F1
        num, den = 2 * grouped["vertex_tp"].sum(), grouped["vertices_gt"].sum() + grouped["vertices_pred"].sum()
    return num.reindex(image_ids, fill_value=0).to_numpy(float), den.reindex(image_ids, fill_value=0).to_numpy(float)


def ratio_terms(run: RunTables, metric: str, image_ids, gt_ids: set[int] | None = None):
    """(numerator, denominator) per image of image_ids (unique), such that the metric of any weights w is
    Σ w · numerator / Σ w · denominator. gt_ids restricts the paired metrics to those GT."""
    image_ids = list(image_ids)
    if metric in PAIRED_METRICS:
        return _paired_terms(run, metric, image_ids, gt_ids)
    t = run.per_image.loc[image_ids]
    ones = np.ones(len(t))
    with_pred = (t["n_pred"] > 0).to_numpy(float)
    with_gt_vertices = (t["n_vertices_gt"] > 0).to_numpy(float)
    ratio = t["n_vertices"].to_numpy(float) / t["n_vertices_gt"].clip(lower=1).to_numpy(float)
    terms = {
        "IoU": (t["inter_px"], t["gt_px"] + t["pred_px"] - t["inter_px"]),
        "IoU_per_image": (t["iou"], ones),
        "F1": (2 * t["inter_px"], t["gt_px"] + t["pred_px"]),
        "BoundaryIoU": (t["biou"], ones),
        "CIoU": (t["ciou"], ones),
        "N_ratio": (ratio * with_gt_vertices, with_gt_vertices),
        "CIoU_p2p": (np.nan_to_num(t["ciou_p2p"].to_numpy(float)) * with_pred, with_pred),
        "PoLiS": (t["polis_image"], (t["polis_image"] != 0).to_numpy(float)),
        "MTA": (t["mta_sum"], t["mta_count"]),
        "recall_inst50": (t["n_matched"], t["n_gt"]),
        "merge_rate": (t["n_merged"], t["n_gt"]),
        "split_rate": (t["n_split"], t["n_gt"]),
    }
    if metric not in terms:
        raise ValueError(f"no ratio terms for metric {metric!r}")
    num, den = terms[metric]
    return np.asarray(num, float), np.asarray(den, float)


def metric_values(run: RunTables, metric: str, image_ids, weights: np.ndarray, gt_ids=None) -> np.ndarray:
    """The metric for every row of weights (n_rows, n_images), the images being image_ids (unique, in that order);
    NaN where a replicate has an empty denominator."""
    weights = np.atleast_2d(weights)
    if metric in AP_METRICS:
        if run.caches is None:
            raise ValueError("AP needs the cached evalImgs of the run (run_eval with cache_evalimgs)")
        counts = weights.astype(np.int64)
        if (counts != weights).any():
            raise ValueError("AP weights must be draw counts (whole numbers)")
        cache = run.caches[AP_METRICS[metric]]
        positions = np.asarray(cache.positions(image_ids))
        return np.array([coco_ap(cache, np.repeat(positions, w)) for w in counts])
    num, den = ratio_terms(run, metric, image_ids, gt_ids)
    numerator, denominator = weights @ num, weights @ den
    with np.errstate(divide="ignore", invalid="ignore"):
        values = numerator / denominator
    return np.where(denominator > 0, values, EMPTY_VALUES.get(metric, math.nan))


def bootstrap_weights(n_images: int, n_rep: int = 1000, seed: int = 0) -> np.ndarray:
    """(n_rep, n_images) draw counts: each replicate draws n_images images with replacement."""
    draws = np.random.default_rng(seed).integers(0, n_images, size=(n_rep, n_images))
    return np.stack([np.bincount(row, minlength=n_images) for row in draws])


def stratified_weights(sizes: list[int], n_rep: int = 1000, seed: int = 0) -> np.ndarray:
    """(n_rep, Σ sizes) draw counts of consecutive groups, each resampled within itself (§4, Δpadat − Δjarang)."""
    rng = np.random.default_rng(seed)
    blocks = []
    for size in sizes:
        draws = rng.integers(0, size, size=(n_rep, size))
        blocks.append(np.stack([np.bincount(row, minlength=size) for row in draws]))
    return np.concatenate(blocks, axis=1)


def _percentiles(values: np.ndarray) -> list[float] | None:
    values = values[~np.isnan(values)]
    return [float(v) for v in np.percentile(values, [2.5, 97.5])] if values.size else None


def _paired_values(a: RunTables, b: RunTables, metric: str, image_ids, weights: np.ndarray):
    gt_ids = common_gt(a, b, metric) if metric in PAIRED_METRICS else None
    return metric_values(a, metric, image_ids, weights, gt_ids), metric_values(b, metric, image_ids, weights, gt_ids)


def _unique(ids) -> list[int]:
    return list(dict.fromkeys(int(i) for i in ids))


def compare(
    runs_a: list[RunTables], runs_b: list[RunTables], metric: str, image_ids, n_rep: int = 1000, seed: int = 0
) -> dict:
    """Δ = metric(B) − metric(A) on image_ids: per seed (runs_a[k] with runs_b[k]) and averaged over seeds, with the
    95% percentile CI of the seed-averaged Δ over cluster-bootstrap replicates shared by all runs (§4)."""
    if len(runs_a) != len(runs_b) or not runs_a:
        raise ValueError("compare needs the same number (≥ 1) of runs, one per seed, for both methods")
    image_ids = _unique(image_ids)
    weights = np.vstack([np.ones((1, len(image_ids)), int), bootstrap_weights(len(image_ids), n_rep, seed)])
    values = [_paired_values(a, b, metric, image_ids, weights) for a, b in zip(runs_a, runs_b, strict=True)]
    deltas = np.stack([vb - va for va, vb in values])  # (seeds, 1 + n_rep); row 0 of the weights is all images
    replicates = deltas[:, 1:].mean(axis=0)
    return {
        "metric": metric,
        "n_images": len(image_ids),
        "n_rep": n_rep,
        "seed": seed,
        "a": [float(va[0]) for va, _ in values],
        "b": [float(vb[0]) for _, vb in values],
        "delta_per_seed": [float(d) for d in deltas[:, 0]],
        "delta": float(deltas[:, 0].mean()),
        "ci95": _percentiles(replicates),
        "nan_replicates": int(np.isnan(replicates).sum()),
    }


def compare_strata(
    runs_a: list[RunTables],
    runs_b: list[RunTables],
    metric: str,
    dense_ids,
    sparse_ids,
    n_rep: int = 1000,
    seed: int = 0,
) -> dict:
    """(Δ in the dense images) − (Δ in the sparse images), with a stratified bootstrap that resamples each group
    within itself (§4); "the effect is largest in dense areas" needs its CI to exclude 0 (EXPERIMENTS §8.4)."""
    if len(runs_a) != len(runs_b) or not runs_a:
        raise ValueError("compare_strata needs the same number (≥ 1) of runs, one per seed, for both methods")
    dense, sparse = _unique(dense_ids), _unique(sparse_ids)
    weights = stratified_weights([len(dense), len(sparse)], n_rep, seed)
    weights = np.vstack([np.ones((1, weights.shape[1]), int), weights])
    w_dense, w_sparse = weights[:, : len(dense)], weights[:, len(dense) :]
    delta_dense, delta_sparse = [], []
    for a, b in zip(runs_a, runs_b, strict=True):
        dense_a, dense_b = _paired_values(a, b, metric, dense, w_dense)
        sparse_a, sparse_b = _paired_values(a, b, metric, sparse, w_sparse)
        delta_dense.append(dense_b - dense_a)
        delta_sparse.append(sparse_b - sparse_a)
    delta_dense, delta_sparse = np.stack(delta_dense), np.stack(delta_sparse)
    difference = (delta_dense - delta_sparse).mean(axis=0)
    return {
        "metric": metric,
        "n_dense": len(dense),
        "n_sparse": len(sparse),
        "n_rep": n_rep,
        "seed": seed,
        "delta_dense": float(delta_dense[:, 0].mean()),
        "delta_sparse": float(delta_sparse[:, 0].mean()),
        "difference": float(difference[0]),
        "difference_per_seed": [float(v) for v in (delta_dense - delta_sparse)[:, 0]],
        "ci95": _percentiles(difference[1:]),
        "nan_replicates": int(np.isnan(difference[1:]).sum()),
    }


def seed_summary(values) -> dict:
    """Mean ± SD over seeds (§4; SD with ddof = 1, None for a single seed)."""
    values = np.asarray(list(values), float)
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if values.size > 1 else None,
        "n": int(values.size),
    }
