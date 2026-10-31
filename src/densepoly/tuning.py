"""Post-processing tuning on val_tune_sub (docs/EXPERIMENTS.md §4), frozen into configs/postproc/<dataset>/<exp>.yaml
before any test (scripts/tune_postproc.py).

Stage 1 tunes the instance stage with polygons in mode raw and keeps the highest AP. Stage 2 tunes the polygon stage
(ρ × k × δ) on the instances of stage 1 and keeps the highest AP or, within tie_ap points of it, the smallest
PoLiS_paired; models with raw polygons have no stage 2. The grid, configs/postproc/grid.yaml, is the same for every
model; stage 1 is τ × m_min × A_min on three-class models, A_min on binary ones and t_min on B0. Networks are tuned on
their saved channels, without running them again. Every grid point is scored with run_eval.tuning_metrics, the AP and
PoLiS_paired of the full evaluation.
"""

from __future__ import annotations

import itertools
import multiprocessing
import time
from dataclasses import replace
from pathlib import Path

from omegaconf import OmegaConf

from densepoly import paths
from densepoly.classical import b0
from densepoly.eval import run_eval
from densepoly.infer import (
    B0Job,
    NetworkJob,
    cached_mbi,
    chunks,
    network_instances,
    network_polygons,
    saved_channels,
)
from densepoly.postproc.polygonize import PolygonParams, polygonize, to_results


def grid_file(root: Path | None = None) -> Path:
    return (paths.configs_dir() if root is None else root) / "postproc" / "grid.yaml"


def load_grid(root: Path | None = None) -> dict:
    return OmegaConf.to_container(OmegaConf.load(grid_file(root)))


def grid_points(axes: dict) -> list[dict]:
    """Every combination of the axes, the last axis varying fastest, in the order of the grid file."""
    names = list(axes)
    return [dict(zip(names, values, strict=True)) for values in itertools.product(*(axes[n] for n in names))]


def choose(rows: list[dict], tie_ap: float | None = None) -> int:
    """Index of the chosen row: the highest AP; with tie_ap (AP points), the smallest PoLiS_paired among the rows
    within tie_ap of it, if any of them has pairs to measure. Ties keep the first row in grid order."""
    best = max(row["AP"] for row in rows)
    first_best = next(i for i, row in enumerate(rows) if row["AP"] == best)
    if tie_ap is None:
        return first_best
    near = [i for i, row in enumerate(rows) if row["AP"] >= best - tie_ap / 100 and row["PoLiS_paired"] is not None]
    return min(near, key=lambda i: (rows[i]["PoLiS_paired"], i)) if near else first_best


def polygon_params(base: PolygonParams, point: dict, mode: str) -> PolygonParams:
    """base with the polygon-stage grid point (rect_iou, dp_k, snap_deg) and the mode."""
    return replace(base, mode=mode, regularize=replace(base.regularize, **point))


def _score(args) -> dict:
    gt, results = args
    return run_eval.tuning_metrics(gt, results)


def score_all(gt: dict, result_sets: list[list[dict]], workers: int) -> list[dict]:
    """tuning_metrics of every result set, in parallel over the sets."""
    if workers > 1 and len(result_sets) > 1:
        with multiprocessing.get_context("spawn").Pool(min(workers, len(result_sets))) as pool:
            return pool.map(_score, [(gt, results) for results in result_sets])
    return [_score((gt, results)) for results in result_sets]


def _b0_stage_chunk(args) -> list[list[dict]]:
    """Results of one chunk of images for every (B0 params, polygon params) of a stage: instances are computed once
    per distinct B0 params."""
    job, settings, image_ids = args
    out = [[] for _ in settings]
    for image_id in image_ids:
        mbi_map = cached_mbi(job.dataset, job.split, image_id)
        instances = {}
        for k, (b0_params, polygon) in enumerate(settings):
            if b0_params not in instances:
                instances[b0_params] = b0.instances(mbi_map, b0_params)
            inst = instances[b0_params]
            out[k] += to_results(polygonize(inst.instance_map, polygon), inst.scores, image_id, job.category)
    return out


def _network_stage_chunk(args) -> list[list[dict]]:
    """As _b0_stage_chunk for (instance params, polygon params) on the saved channels of a network."""
    job, settings, image_ids = args
    out = [[] for _ in settings]
    for image_id in image_ids:
        channels = saved_channels(job, image_id)
        instances = {}
        for k, (instance, polygon) in enumerate(settings):
            if instance not in instances:
                instances[instance] = network_instances(channels, instance)
            found = instances[instance]
            polygons = network_polygons(found, channels, polygon)
            out[k] += to_results(polygons, found.scores, image_id, job.category)
    return out


def stage_results(chunk_function, job, settings: list, image_ids: list[int], workers: int) -> list[list[dict]]:
    """One result set per setting on the images, in parallel over chunks of images."""
    parts = [(job, settings, part) for part in chunks(image_ids, max(1, workers) * 4)]
    if workers > 1:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            done = pool.map(chunk_function, parts)
    else:
        done = [chunk_function(part) for part in parts]
    return [[r for part in done for r in part[k]] for k in range(len(settings))]


def tune_b0(job: B0Job, gt: dict, grid: dict, workers: int = 1) -> dict:
    """Both stages for B0 on the images of gt; returns the chosen values and the score of every grid point."""
    image_ids = [image["id"] for image in gt["images"]]
    raw = replace(job.polygon, mode="raw")
    tic = time.perf_counter()
    stage1_points = grid_points(grid["b0"])
    settings = [(replace(job.b0, **point), raw) for point in stage1_points]
    stage1 = score_all(gt, stage_results(_b0_stage_chunk, job, settings, image_ids, workers), workers)
    first = choose(stage1)
    chosen_b0 = settings[first][0]
    stage1_seconds = time.perf_counter() - tic

    tic = time.perf_counter()
    stage2_points = grid_points(grid["polygon"])
    settings = [(chosen_b0, polygon_params(job.polygon, point, job.polygon.mode)) for point in stage2_points]
    stage2 = score_all(gt, stage_results(_b0_stage_chunk, job, settings, image_ids, workers), workers)
    second = choose(stage2, grid["tie_ap"])
    return {
        "chosen": {"b0": stage1_points[first], **stage2_points[second]},
        "score": stage2[second],
        "stage1": [{**point, **score} for point, score in zip(stage1_points, stage1, strict=True)],
        "stage2": [{**point, **score} for point, score in zip(stage2_points, stage2, strict=True)],
        "seconds": {"stage1": round(stage1_seconds, 1), "stage2": round(time.perf_counter() - tic, 1)},
        "images": len(image_ids),
    }


def tune_network(job: NetworkJob, gt: dict, grid: dict, binary: bool, workers: int = 1) -> dict:
    """Both stages for a network on its saved channels of the images of gt, as tune_b0; stage 2 only for regularised
    polygons."""
    image_ids = [image["id"] for image in gt["images"]]
    raw = replace(job.polygon, mode="raw")
    tic = time.perf_counter()
    stage1_points = grid_points(grid["binary" if binary else "instance"])
    settings = [(replace(job.instance, **point), raw) for point in stage1_points]
    stage1 = score_all(gt, stage_results(_network_stage_chunk, job, settings, image_ids, workers), workers)
    first = choose(stage1)
    chosen_instance = settings[first][0]
    seconds = {"stage1": round(time.perf_counter() - tic, 1)}
    report = {
        "chosen": stage1_points[first],
        "score": stage1[first],
        "stage1": [{**point, **score} for point, score in zip(stage1_points, stage1, strict=True)],
        "stage2": [],
        "seconds": seconds,
        "images": len(image_ids),
    }
    if job.polygon.mode == "raw":
        return report
    tic = time.perf_counter()
    stage2_points = grid_points(grid["polygon"])
    settings = [(chosen_instance, polygon_params(job.polygon, point, job.polygon.mode)) for point in stage2_points]
    stage2 = score_all(gt, stage_results(_network_stage_chunk, job, settings, image_ids, workers), workers)
    second = choose(stage2, grid["tie_ap"])
    seconds["stage2"] = round(time.perf_counter() - tic, 1)
    return {
        **report,
        "chosen": {**stage1_points[first], **stage2_points[second]},
        "score": stage2[second],
        "stage2": [{**point, **score} for point, score in zip(stage2_points, stage2, strict=True)],
    }
