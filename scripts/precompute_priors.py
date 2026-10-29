"""Morphological priors (docs/METHOD.md §2, docs/DATA.md §5): measure the cost, freeze the scaling, then cache.

    python scripts/precompute_priors.py --data crowdai --benchmark 50      # time 50 train units; writes nothing
    python scripts/precompute_priors.py --data crowdai --stats             # q99.5 and mean/std from 500 train units
    python scripts/precompute_priors.py --data crowdai --split train_dedup # cache a split (ask before full runs)

A unit is a CrowdAI image or an official WHU tile (densepoly.priors.cache). Caching resumes: units whose PNG exists
are skipped. --stats writes configs/data/<dataset>_prior_stats.json, which caching needs and which is frozen once
the cache is built; cache timings go to splits/<dataset>_prior_cache.json.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from densepoly import paths
from densepoly.config import load_config
from densepoly.priors import cache
from densepoly.utils.io import read_json, write_json
from densepoly.utils.provenance import git_revision, timestamp

SPLITS = {"crowdai": ("train_dedup", "val_tune", "test", "val_clean"), "whu": ("train_dedup", "val_tune", "test")}


def priors_section(dataset: str) -> dict:
    """The priors config of a dataset (the same for every experiment)."""
    return OmegaConf.to_container(load_config(dataset, "full_p2").priors)


def pool_map(function, jobs: list[tuple], workers: int) -> list:
    if workers > 1 and len(jobs) > 1:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            return pool.starmap(function, jobs, chunksize=1)
    return [function(*job) for job in jobs]


def sample(units: list[cache.Unit], n: int, seed: int = 0) -> list[cache.Unit]:
    picked = np.random.default_rng(seed).permutation(len(units))[:n]
    return [units[i] for i in sorted(picked)]


def benchmark(dataset: str, n: int, workers: int, splits_dir: Path | None) -> dict:
    priors = priors_section(dataset)
    units = sample(cache.units(dataset, "train_dedup", splits_dir), n)
    start = time.perf_counter()
    seconds = [s for _, s in pool_map(cache.sample_unit, [(u, priors, 1, 0) for u in units], workers)]
    wall = time.perf_counter() - start
    totals = {split: len(cache.units(dataset, split, splits_dir)) for split in SPLITS[dataset]}
    per_unit = float(np.mean(seconds))
    return {
        "units_timed": len(units),
        "workers": workers,
        "seconds_per_unit_one_core": round(per_unit, 3),
        "wall_seconds": round(wall, 1),
        "units_per_split": totals,
        "projected_core_hours": round(per_unit * sum(totals.values()) / 3600, 2),
        "projected_wall_hours_at_measured_rate": round(wall / len(units) * sum(totals.values()) / 3600, 2),
        "priors": priors,
    }


def write_stats(dataset: str, workers: int, splits_dir: Path | None) -> dict:
    priors = priors_section(dataset)
    units = sample(cache.units(dataset, "train_dedup", splits_dir), cache.STATS_UNITS)
    jobs = [(u, priors, cache.STATS_PIXELS_PER_UNIT, i) for i, u in enumerate(units)]
    results = pool_map(cache.sample_unit, jobs, workers)
    record = {
        "dataset": dataset,
        "created": timestamp(),
        "git": git_revision(),
        "units": len(units),
        "pixels_per_unit": cache.STATS_PIXELS_PER_UNIT,
        "percentile": cache.PERCENTILE,
        "priors": priors,
        **cache.statistics(np.concatenate([r[0] for r in results])),
        "seconds_per_unit": round(float(np.mean([r[1] for r in results])), 3),
    }
    write_json(cache.stats_file(dataset), record)
    return record


def write_cache(dataset: str, split: str, workers: int, splits_dir: Path | None, out: Path) -> dict:
    stats = cache.load_stats(dataset)
    priors = priors_section(dataset)
    if stats["priors"] != priors:
        raise SystemExit(f"{cache.stats_file(dataset)} was made with other prior settings; rerun --stats first")
    units = cache.units(dataset, split, splits_dir)
    todo = [u for u in units if not paths.prior_cache(dataset, split, u.key).exists()]
    jobs = [(u, paths.prior_cache(dataset, split, u.key), priors, stats["q"]) for u in todo]
    start = time.perf_counter()
    seconds = pool_map(cache.cache_unit, jobs, workers)
    record = {
        "created": timestamp(),
        "git": git_revision(),
        "units": len(units),
        "computed_now": len(todo),
        "already_cached": len(units) - len(todo),
        "workers": workers,
        "wall_seconds": round(time.perf_counter() - start, 1),
        "core_seconds": round(float(np.sum(seconds)), 1),
        "seconds_per_unit": round(float(np.mean(seconds)), 3) if seconds else None,
    }
    timings_file = out / f"{dataset}_prior_cache.json"
    timings = read_json(timings_file) if timings_file.exists() else {}
    timings[split] = record
    write_json(timings_file, timings)
    return record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", choices=sorted(SPLITS), required=True)
    parser.add_argument("--split", choices=sorted({s for v in SPLITS.values() for s in v}), help="split to cache")
    parser.add_argument("--benchmark", type=int, metavar="N", help="time N random train units, write nothing")
    parser.add_argument("--stats", action="store_true", help="freeze the scaling statistics from 500 train units")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--splits-dir", type=Path, help="folder of split files (default: repo splits/)")
    args = parser.parse_args(argv)
    if args.split and args.split not in SPLITS[args.data]:
        parser.error(f"--split for {args.data} must be one of {SPLITS[args.data]}")
    if not (args.benchmark or args.stats or args.split):
        parser.error("nothing to do: give --benchmark N, --stats or --split")

    if args.benchmark:
        print(json.dumps(benchmark(args.data, args.benchmark, args.workers, args.splits_dir), indent=1))
    if args.stats:
        print(json.dumps(write_stats(args.data, args.workers, args.splits_dir), indent=1))
    if args.split:
        out = args.splits_dir or paths.splits_dir()
        print(json.dumps(write_cache(args.data, args.split, args.workers, args.splits_dir, out), indent=1))


if __name__ == "__main__":
    main()
