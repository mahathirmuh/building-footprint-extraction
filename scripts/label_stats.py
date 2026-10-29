"""Label statistics per split (PLAN.md Phase 2): the shares of building pixels, of contact pixels among building pixels,
and of building pixels with a θ target (V), from the cached instance maps and the dataset's label parameters.

    python scripts/label_stats.py --data crowdai --split train_dedup val_tune test

Writes splits/<dataset>_label_stats.json (one entry per split).
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
from collections import Counter
from pathlib import Path

from densepoly import paths
from densepoly.config import load_config
from densepoly.labels.contact import class_map
from densepoly.labels.orientation import orientation_targets
from densepoly.utils.io import read_json, read_png, write_json
from densepoly.utils.provenance import git_revision, timestamp

SPLITS = ("train_dedup", "val_tune", "test", "val_clean")


def tile_counts(path: Path, radius: int, min_area: int) -> Counter:
    instance_map = read_png(path)
    classes = class_map(instance_map, radius)
    orientation = orientation_targets(instance_map, min_area)
    contact = int((classes == 2).sum())
    return Counter(
        tiles=1,
        pixels=instance_map.size,
        building_px=int((instance_map > 0).sum()),
        contact_px=contact,
        theta_valid_px=int(orientation.valid.sum()),
        instances=len(set(instance_map[instance_map > 0].tolist())),
        instances_with_theta=len(orientation.theta),
        tiles_with_contact=int(contact > 0),
    )


def split_counts(dataset: str, split: str, radius: int, min_area: int, workers: int) -> Counter:
    with (paths.instance_dir(dataset, split) / "index.csv").open(encoding="utf-8") as f:
        files = [paths.instance_cache(dataset, split, row["image_id"]) for row in csv.DictReader(f)]
    jobs = [(path, radius, min_area) for path in files]
    if workers > 1:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            results = pool.starmap(tile_counts, jobs, chunksize=64)
    else:
        results = [tile_counts(*job) for job in jobs]
    return sum(results, Counter())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", choices=paths.DATASETS, required=True)
    parser.add_argument("--split", nargs="+", choices=SPLITS, required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--splits-dir", type=Path, help="folder for the statistics file (default: repo splits/)")
    args = parser.parse_args(argv)

    labels = load_config(args.data, "full_p2").labels
    radius, min_area = int(labels.contact_radius_px), int(labels.theta_min_area_px)
    stats_path = (args.splits_dir or paths.splits_dir()) / f"{args.data}_label_stats.json"
    stats = read_json(stats_path) if stats_path.exists() else {}
    for split in args.split:
        c = split_counts(args.data, split, radius, min_area, args.workers)
        stats[split] = {
            "created": timestamp(),
            "git": git_revision(),
            "contact_radius_px": radius,
            "theta_min_area_px": min_area,
            **dict(c),
            "building_share_of_pixels": round(c["building_px"] / max(c["pixels"], 1), 5),
            "contact_share_of_building_px": round(c["contact_px"] / max(c["building_px"], 1), 5),
            "theta_valid_share_of_building_px": round(c["theta_valid_px"] / max(c["building_px"], 1), 5),
            "share_of_tiles_with_contact": round(c["tiles_with_contact"] / max(c["tiles"], 1), 5),
        }
    write_json(stats_path, stats)
    print(json.dumps({s: stats[s] for s in args.split}, indent=1))


if __name__ == "__main__":
    main()
