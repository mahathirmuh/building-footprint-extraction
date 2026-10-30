"""One evaluation: the predictions of one method on one split against its ground truth (docs/EVALUATION.md §1–5).

Writes outputs/metrics/<dataset>/<exp>/<run>/<split>[_<variant>].json (schema §5) and, next to it for the statistics,
<name>.per_image.parquet, <name>.pairs.parquet and <name>.evalimgs.pkl (the cached COCOeval results of AP and
AP_boundary). The metrics of §2.1 come from:
- AP: pycocotools COCOeval "segm"; AP_boundary: the vendored boundary-iou-api COCOeval "boundary". evaluate() runs
  once and eval/stats.py accumulates any list of images (all, a stratum, a bootstrap resample);
- pixel metrics and the main C-IoU: eval/pixel.py; instance and paired metrics: eval/instance.py;
- CIoU_p2p, PoLiS and MTA: the vendored Pix2Poly scripts, which also return per-image values. Strata pool those
  values again with each script's rule; on all images the result is checked against the script's own pooled value.
Ratios (AP, IoU, C-IoU, rates) are fractions in [0, 1], PoLiS is in pixels, MTA and θ errors are in degrees.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import math
import pickle
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import multiprocess
import numpy as np
import pandas as pd
from boundary_iou.coco_instance_api.coco import COCO as BoundaryCOCO
from boundary_iou.coco_instance_api.cocoeval import COCOeval as BoundaryCOCOeval
from boundary_iou.utils.boundary_utils import augment_annotations_with_boundary_single_core
from pix2poly_metrics import angle_eval, cIoU, polis
from pycocotools.cocoeval import COCOeval

from densepoly import paths
from densepoly.eval import coco_io, instance, pixel, stats
from densepoly.utils.io import read_json, write_json
from densepoly.utils.provenance import git_revision, timestamp

STRATA_MAIN = ("jarang", "sedang", "padat")
EMPTY_STRATUM = "kosong"
STRATA_SECONDARY = ("bersentuhan", "tidak_bersentuhan")


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def load_strata(dataset: str, split: str, image_ids, splits_dir: Path | None = None) -> dict[str, list[int]]:
    """{stratum: image ids of the split} from the frozen density file, main strata and the secondary index (docs/
    DATA.md §4); val_tune_sub and val_select use the file of val_tune. {} without a density file (val_full)."""
    file = paths.density_file(dataset, paths.cache_split(split), splits_dir)
    if not file.exists():
        return {}
    density = read_json(file)
    main, secondary = density["main"]["strata"], density["secondary"]["strata"]
    if sorted(main[EMPTY_STRATUM]) != sorted(secondary[EMPTY_STRATUM]):
        raise ValueError(f"{file}: the empty stratum differs between the two indices")
    keep = set(image_ids)
    return {name: [i for i in ids if i in keep] for name, ids in {**main, **secondary}.items()}


def segm_evaluation(gt: dict, results: list[dict]) -> COCOeval:
    """pycocotools COCOeval of mask AP after evaluate()."""
    coco_gt = coco_io.coco_from_dict(gt)
    evaluator = COCOeval(coco_gt, coco_io.load_results(coco_gt, results), "segm")
    with _quiet():
        evaluator.evaluate()
    return evaluator


def boundary_evaluation(gt: dict, results: list[dict], dilation_ratio: float = pixel.DILATION_RATIO):
    """Vendored boundary-iou-api COCOeval "boundary" after evaluate(). The boundaries are added with its single-core
    helper (same computation as its multi-core one, which pickles the whole COCO object for every worker)."""
    with _quiet():
        coco_gt = BoundaryCOCO()
        coco_gt.dataset = copy.deepcopy(gt)
        coco_gt.createIndex()
        if results:
            coco_dt = coco_gt.loadRes(copy.deepcopy(results))
        else:
            coco_dt = BoundaryCOCO()
            coco_dt.dataset = {"images": list(gt["images"]), "categories": copy.deepcopy(gt["categories"])}
            coco_dt.dataset["annotations"] = []
            coco_dt.createIndex()
        for coco in (coco_gt, coco_dt):
            # The annotation dicts are updated in place, so the index already sees their "boundary".
            augment_annotations_with_boundary_single_core(
                0, coco.dataset["annotations"], coco.annToMask, dilation_ratio
            )
            coco.get_boundary, coco.dilation_ratio = True, dilation_ratio
        evaluator = BoundaryCOCOeval(coco_gt, coco_dt, iouType="boundary", dilation_ratio=dilation_ratio)
        evaluator.evaluate()
    return evaluator


class _ChunkedPool:
    """Pool for ContourEval, which calls pool.imap(function, items) and would otherwise send one image per task."""

    def __init__(self, pool, chunksize: int):
        self.pool, self.chunksize = pool, chunksize

    def imap(self, function, items):
        return self.pool.imap(function, items, chunksize=self.chunksize)


def vendor_metrics(gt_path: Path, gt: dict, results: list[dict], workers: int = 1) -> dict:
    """Per-image values and pooled numbers of the vendored CIoU_p2p (cIoU.py), PoLiS (polis.py) and MTA
    (angle_eval.py), each on its own COCO objects."""
    multi_ring = sum(1 for r in results if len(r["segmentation"]) > 1)
    if multi_ring:
        raise ValueError(
            f"{multi_ring} predictions have several rings: the vendored cIoU.py reads one ring per prediction "
            "(its prediction mask is reshaped to (H, W)), so these predictions cannot be evaluated as they are"
        )
    if results:
        with tempfile.TemporaryDirectory() as tmp:
            predictions = Path(tmp) / "predictions.json"
            predictions.write_text(json.dumps(results), encoding="utf-8")
            with _quiet():
                ciou = cIoU.compute_IoU_cIoU(str(predictions), str(gt_path), progress=False)
    else:  # cIoU.py only iterates images with predictions
        ciou = {"iou": None, "ciou": None, "n_ratio": None, "n_images": 0, "per_image": {}}

    # polis.py and angle_eval.py build shapely polygons; released rings such as [a, b, a] stay out of them only.
    shapely_results = coco_io.shapely_readable(results)
    coco_gt = coco_io.coco_from_dict(gt)
    polis_eval = polis.PolisEval(coco_gt, coco_io.load_results(coco_gt, shapely_results))
    with _quiet():
        polis_value = polis_eval.evaluate()

    coco_gt = coco_io.coco_from_dict(gt)
    contour = angle_eval.ContourEval(coco_gt, coco_io.load_results(coco_gt, shapely_results))
    with _quiet():
        if workers > 1 and len(gt["images"]) > 1:
            with multiprocess.get_context("spawn").Pool(workers) as pool:
                chunksize = max(1, len(gt["images"]) // (8 * workers))
                mta = contour.evaluate_per_image(_ChunkedPool(pool, chunksize))
        else:
            mta = contour.evaluate_per_image()
    values = [v for img in contour.img_ids for v in mta[img]]
    return {
        "ciou_per_image": {img: v["ciou"] for img, v in ciou["per_image"].items()},
        "IoU_p2p": ciou["iou"],
        "CIoU_p2p": ciou["ciou"],
        "N_ratio_p2p": ciou["n_ratio"],
        "polis_per_image": dict(polis_eval.per_image),
        "PoLiS": polis_value,
        "PoLiS_valid_images": polis_eval.num_valid_imgs,
        "mta_per_image": {img: (float(np.sum(v)), len(v)) for img, v in mta.items()},
        "MTA": float(np.mean(values)) if values else None,
        "MTA_values": len(values),
        "excluded_from_polis_mta": len(results) - len(shapely_results),
    }


def _mean(values) -> float | None:
    return float(np.mean(values)) if len(values) else None


def pooled_vendor(rows: pd.DataFrame) -> dict:
    """IoU_p2p, CIoU_p2p, N_ratio_p2p, PoLiS and MTA of a set of per-image rows, with the pooling rules of the
    vendored scripts: IoU, C-IoU and N-ratio over images with predictions (cIoU.py; its IoU is the "IoU" HiSup
    reports), PoLiS over images whose value is not 0, MTA over all values."""
    with_pred = rows[rows["n_pred"] > 0]
    has_gt_vertices = with_pred[with_pred["n_vertices_gt"] > 0]
    polis_valid = rows.loc[rows["polis_image"] != 0, "polis_image"]
    mta_count = int(rows["mta_count"].sum())
    return {
        "IoU_p2p": _mean(with_pred["iou"].to_numpy()),
        "CIoU_p2p": _mean(with_pred["ciou_p2p"].to_numpy()),
        "N_ratio_p2p": _mean((has_gt_vertices["n_vertices"] / has_gt_vertices["n_vertices_gt"]).to_numpy()),
        "PoLiS": _mean(polis_valid.to_numpy()),
        "PoLiS_valid_images": int(polis_valid.size),
        "MTA": float(rows["mta_sum"].sum() / mta_count) if mta_count else None,
        "MTA_values": mta_count,
    }


def _check_pooling(vendor: dict, pooled: dict) -> None:
    """The pooling of per-image values must reproduce the vendored scripts' own numbers on all images."""
    for key in ("IoU_p2p", "CIoU_p2p", "N_ratio_p2p", "PoLiS", "MTA", "PoLiS_valid_images", "MTA_values"):
        a, b = vendor[key], pooled[key]
        if (a is None) != (b is None) or (a is not None and not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)):
            raise RuntimeError(f"pooled {key} = {b} differs from the vendored script ({a})")


def per_image_table(pixel_rows, gt_rows, vendor: dict, strata: dict[str, list[int]]) -> pd.DataFrame:
    """One row per image (docs/EVALUATION.md §5): pixel values, vendored values and instance counts."""
    counts = defaultdict(lambda: {"n_matched": 0, "n_merged": 0, "n_split": 0})
    for row in gt_rows:
        c = counts[row["image_id"]]
        c["n_matched"] += row["matched"]
        c["n_merged"] += row["merged"]
        c["n_split"] += row["split"]
    main = {i: name for name in (*STRATA_MAIN, EMPTY_STRATUM) for i in strata.get(name, [])}
    secondary = {i: name for name in (*STRATA_SECONDARY, EMPTY_STRATUM) for i in strata.get(name, [])}
    rows = []
    for row in pixel_rows:
        img = row["image_id"]
        mta_sum, mta_count = vendor["mta_per_image"].get(img, (0.0, 0))
        rows.append(
            {
                **row,
                "stratum": main.get(img, ""),
                "stratum_secondary": secondary.get(img, ""),
                "ciou_p2p": vendor["ciou_per_image"].get(img, math.nan),
                "polis_image": vendor["polis_per_image"].get(img, 0.0),
                "mta_sum": mta_sum,
                "mta_count": mta_count,
                **counts[img],
            }
        )
    return pd.DataFrame(rows)


def summarize(
    image_ids: list[int],
    table: pd.DataFrame,
    gt_rows: list[dict],
    pair_rows: list[dict],
    caches: dict[str, stats.EvalCache],
    n_fallback: int | None = None,
) -> dict:
    """All metrics of §2.1 over a set of images (all of a split, or one stratum)."""
    keep = set(image_ids)
    rows = table[table["image_id"].isin(keep)]
    segm = stats.coco_stats(caches["segm"], caches["segm"].positions(image_ids))
    boundary = stats.coco_stats(caches["boundary"], caches["boundary"].positions(image_ids))
    pixels = pixel.summarize_pixel(rows.to_dict("records"))
    instances = instance.summarize_instance(
        [r for r in gt_rows if r["image_id"] in keep], [p for p in pair_rows if p["image_id"] in keep]
    )
    summary = {
        "AP": segm["AP"],
        "AP50": segm["AP50"],
        "AP75": segm["AP75"],
        "AP_boundary": boundary["AP"],
        **{key: pixels[key] for key in ("IoU", "IoU_per_image", "F1", "BoundaryIoU", "CIoU", "N_ratio")},
        **pooled_vendor(rows),
        **{key: instances[key] for key in ("PoLiS_paired", "MTA_inst", "merge_rate", "split_rate", "recall_inst50")},
        **{key: instances[key] for key in ("vertex_F1", "theta_err_median", "theta_err_p90")},
        "n_images": int(len(rows)),
        "n_images_with_pred": pixels["n_images_with_pred"],
        "n_pred": int(rows["n_pred"].sum()),
        "n_gt": instances["n_gt"],
        "n_pairs": instances["n_pairs"],
        "n_pairs_without_mta": instances["n_pairs_without_mta"],
    }
    if n_fallback is not None:
        summary["n_fallback"] = n_fallback
    return summary


def summarize_empty(image_ids: list[int], table: pd.DataFrame) -> dict:
    """The empty stratum: pixel IoU and false positives per image only (docs/EVALUATION.md §3)."""
    rows = table[table["image_id"].isin(set(image_ids))]
    return {
        "IoU": pixel.summarize_pixel(rows.to_dict("records"))["IoU"],
        "fp_per_image": float(rows["n_pred"].mean()) if len(rows) else None,
        "n_images": int(len(rows)),
    }


def tuning_metrics(gt: dict, results: list[dict]) -> dict:
    """AP and PoLiS_paired, the two numbers the post-processing tuning chooses with (docs/EXPERIMENTS.md §4), from the
    code of evaluate() but without the other metrics and in one process: the tuning runs grid points in parallel."""
    kept, _ = coco_io.check_predictions(results, gt, strict=True)
    ap = stats.coco_stats(stats.eval_cache(segm_evaluation(gt, kept)))["AP"]
    gt_rows, pair_rows = instance.instance_metrics(gt, kept)
    return {"AP": ap, "PoLiS_paired": instance.summarize_instance(gt_rows, pair_rows)["PoLiS_paired"]}


def _relative(path: Path | None) -> str | None:
    if path is None:
        return None
    for root in (paths.out_root(), paths.data_root(), paths.REPO_ROOT):
        with contextlib.suppress(ValueError):
            return Path(path).resolve().relative_to(root).as_posix()
    return str(path)


def evaluate(
    dataset: str,
    split: str,
    exp: str,
    run: str,
    results: list[dict],
    variant: str | None = None,
    released: bool = False,
    workers: int = 1,
    meta: dict | None = None,
    pred_file: Path | None = None,
    cache_evalimgs: bool = True,
    splits_dir: Path | None = None,
    gt_path: Path | None = None,
) -> dict:
    """Evaluate results on paths.gt_file(dataset, split, variant), write the metrics and tables, return the record.
    released: check the predictions leniently (predictions released by another paper). gt_path: another ground truth
    of the same images (the sanity checks of docs/EVALUATION.md §6.1 drop its empty masks)."""
    meta = meta or {}
    seconds = {}
    start = time.perf_counter()
    gt_path = gt_path or paths.gt_file(dataset, split, variant)
    gt = read_json(gt_path)
    kept, check = coco_io.check_predictions(results, gt, strict=not released)
    image_ids = [img["id"] for img in gt["images"]]
    strata = load_strata(dataset, split, image_ids, splits_dir)
    seconds["check"] = time.perf_counter() - start

    def timed(name, function, *args, **kwargs):
        tic = time.perf_counter()
        value = function(*args, **kwargs)
        seconds[name] = time.perf_counter() - tic
        return value

    caches = {
        "segm": stats.eval_cache(timed("ap_segm", segm_evaluation, gt, kept)),
        "boundary": stats.eval_cache(timed("ap_boundary", boundary_evaluation, gt, kept)),
    }
    pixel_rows = timed("pixel", pixel.pixel_metrics, gt, kept)
    gt_rows, pair_rows = timed("instance", instance.instance_metrics, gt, kept, workers)
    vendor = timed("vendor", vendor_metrics, gt_path, gt, kept, workers)
    table = per_image_table(pixel_rows, gt_rows, vendor, strata)
    _check_pooling(vendor, pooled_vendor(table))

    tic = time.perf_counter()
    n_fallback = sum(1 for r in kept if r.get("fallback"))
    overall = summarize(image_ids, table, gt_rows, pair_rows, caches, n_fallback)
    per_stratum = {
        name: summarize(strata[name], table, gt_rows, pair_rows, caches)
        for name in (*STRATA_MAIN, *STRATA_SECONDARY)
        if strata.get(name)
    }
    if strata.get(EMPTY_STRATUM):
        per_stratum[EMPTY_STRATUM] = summarize_empty(strata[EMPTY_STRATUM], table)
    seconds["summaries"] = time.perf_counter() - tic

    record = {
        "dataset": dataset,
        "split": split,
        "variant": variant,
        "exp": exp,
        "run": run,
        "git": git_revision(),
        "created": timestamp(),
        "gt_file": _relative(gt_path),
        "pred_file": _relative(pred_file),
        "checkpoint": meta.get("checkpoint"),
        "postproc": meta.get("postproc"),
        "pred_check": {
            "mode": "lenient" if released else "strict",
            **check,
            "excluded_from_polis_mta": vendor["excluded_from_polis_mta"],
        },
        "overall": overall,
        "coco_stats": {name: stats.coco_stats(cache) for name, cache in caches.items()},
        "per_stratum": per_stratum,
        "efficiency": meta.get("efficiency"),
        "seconds": {key: round(value, 1) for key, value in seconds.items()},
    }
    write_json(paths.metrics_file(dataset, exp, run, split, variant), record)
    table.to_parquet(paths.metrics_artifact(dataset, exp, run, split, "per_image.parquet", variant), index=False)
    pd.DataFrame(pair_rows, columns=_PAIR_COLUMNS).to_parquet(
        paths.metrics_artifact(dataset, exp, run, split, "pairs.parquet", variant), index=False
    )
    if cache_evalimgs:
        with paths.metrics_artifact(dataset, exp, run, split, "evalimgs.pkl", variant).open("wb") as f:
            pickle.dump(caches, f, protocol=pickle.HIGHEST_PROTOCOL)
    return record


_PAIR_COLUMNS = [
    "image_id",
    "gt_id",
    "pred_index",
    "score",
    "iou",
    "polis",
    "mta",
    "theta_err",
    "vertices_gt",
    "vertices_pred",
    "vertex_tp",
]
