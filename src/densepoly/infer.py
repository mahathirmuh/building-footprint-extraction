"""Predictions of an experiment on a split (scripts/predict.py; docs/EVALUATION.md §1).

Phase 4 runs the classical baseline B0 only. Network inference (padding to a multiple of 32, softmax, cropping back,
docs/METHOD.md §6.2 step 1) comes with the first model in Phase 5.
"""

from __future__ import annotations

import multiprocessing
import time
from dataclasses import dataclass

import numpy as np

from densepoly import paths
from densepoly.classical import b0
from densepoly.postproc.polygonize import PolygonParams, polygonize, to_results
from densepoly.priors.cache import load_stats, quantize, read_prior, read_rgb
from densepoly.priors.mbi import brightness, mbi


def cached_mbi(dataset: str, split: str, image_id: int) -> np.ndarray:
    """MBI of one image scaled to [0, 1] (docs/METHOD.md §2.4): channel R of its prior cache divided by 255."""
    if dataset != "crowdai":
        raise NotImplementedError("B0 runs on CrowdAI only (Phase 4); WHU priors are cached per official tile")
    return read_prior(paths.prior_cache(dataset, split, image_id))[..., 0].astype(np.float32) / 255.0


@dataclass(frozen=True)
class B0Job:
    dataset: str
    split: str
    b0: b0.B0Params
    polygon: PolygonParams
    category: int


def b0_image(job: B0Job, image_id: int) -> tuple[list[dict], dict]:
    """Results of B0 on one image, and the seconds of its instance and polygon stages."""
    tic = time.perf_counter()
    instances = b0.instances(cached_mbi(job.dataset, job.split, image_id), job.b0)
    mid = time.perf_counter()
    results = to_results(polygonize(instances.instance_map, job.polygon), instances.scores, image_id, job.category)
    return results, {"instances": mid - tic, "polygons": time.perf_counter() - mid}


def _b0_chunk(args) -> list[tuple[list[dict], dict]]:
    job, image_ids = args
    return [b0_image(job, image_id) for image_id in image_ids]


def chunks(items: list, n: int) -> list[list]:
    """n contiguous chunks of about equal size, so that concatenating their outputs keeps the order of items."""
    bounds = np.linspace(0, len(items), n + 1).astype(int)
    return [items[a:b] for a, b in zip(bounds[:-1], bounds[1:], strict=True) if b > a]


def predict_b0(job: B0Job, image_ids: list[int], workers: int = 1) -> tuple[list[dict], list[dict]]:
    """(results in image order, timings per image) of B0 on a split."""
    parts = chunks(image_ids, max(1, min(workers, len(image_ids))) * 4)
    if workers > 1:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            done = pool.map(_b0_chunk, [(job, part) for part in parts])
    else:
        done = [_b0_chunk((job, part)) for part in parts]
    per_image = [item for part in done for item in part]
    return [r for results, _ in per_image for r in results], [timing for _, timing in per_image]


def b0_seconds_from_rgb(job: B0Job, images: list[dict], mbi_scales, n: int, seed: int = 0) -> dict:
    """Seconds per image of B0 in one process from the RGB tile, MBI included (reading the file excluded), on n random
    images of the split; also checks that the MBI computed again equals the cached one."""
    q = load_stats(job.dataset)["q"][0]
    picked = np.random.default_rng(seed).choice(len(images), size=min(n, len(images)), replace=False)
    seconds, mbi_seconds, same = [], [], 0
    for i in sorted(picked.tolist()):
        image = images[i]
        rgb = read_rgb(paths.image_dir(job.dataset, job.split) / image["file_name"])
        tic = time.perf_counter()
        scaled = quantize(mbi(brightness(rgb), list(mbi_scales))[..., None], [q])[..., 0].astype(np.float32) / 255.0
        mid = time.perf_counter()
        instances = b0.instances(scaled, job.b0)
        to_results(polygonize(instances.instance_map, job.polygon), instances.scores, image["id"], job.category)
        seconds.append(time.perf_counter() - tic)
        mbi_seconds.append(mid - tic)
        same += np.array_equal(scaled, cached_mbi(job.dataset, job.split, image["id"]))
    return {
        "images": len(seconds),
        "seconds_per_image_median": float(np.median(seconds)),
        "seconds_per_image_mean": float(np.mean(seconds)),
        "mbi_seconds_per_image_median": float(np.median(mbi_seconds)),
        "processes": 1,
        "includes": "MBI from RGB, B0 instances and polygons; not reading the file",
        "mbi_equal_to_cache": f"{same}/{len(seconds)}",
    }
