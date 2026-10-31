"""Tune the post-processing of one experiment on val_tune_sub and freeze it (docs/EXPERIMENTS.md §4; densepoly.tuning).

    python scripts/tune_postproc.py --data crowdai --exp b0 --workers 12
    python scripts/tune_postproc.py --data crowdai --exp b4 --workers 12
    python scripts/tune_postproc.py --data crowdai --exp b0 --workers 12 --limit 100 --dry-run   # time a sample

A network is tuned on the channels saved by scripts/predict.py --split val_tune_sub --save-channels, on seed 0; seeds
1-2 reuse the result (docs/EXPERIMENTS.md §4). Writes configs/postproc/<data>/<exp>.yaml: the chosen values, which
load_config merges into the experiment, and how they were chosen (score of every grid point, sha256 of the grid file,
commit, date). An existing file is kept unless --force is given: a frozen tuning must not change once the test was
touched. --dry-run writes nothing and projects the time of the whole split from --limit images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from omegaconf import OmegaConf

from densepoly import infer, paths, tuning
from densepoly.classical.b0 import B0Params
from densepoly.config import load_config, postproc_file
from densepoly.eval import coco_io
from densepoly.postproc.polygonize import PolygonParams
from densepoly.utils.provenance import git_revision, timestamp

SPLIT = "val_tune_sub"
SEED = 0


def first_images(gt: dict, n: int) -> dict:
    keep = {image["id"] for image in gt["images"][:n]}
    return {**gt, "images": gt["images"][:n], "annotations": [a for a in gt["annotations"] if a["image_id"] in keep]}


def frozen_document(args, report: dict, grid_path: Path) -> dict:
    grid_sha256 = hashlib.sha256(grid_path.read_bytes()).hexdigest()
    return {
        "postproc": report["chosen"],
        "tuning": {
            "split": SPLIT,
            "images": report["images"],
            "created": timestamp(),
            "git": git_revision(),
            "grid": {"file": "configs/postproc/grid.yaml", "sha256": grid_sha256},
            "score": report["score"],
            "stage1": report["stage1"],
            "stage2": report["stage2"],
            "seconds": report["seconds"],
            "workers": args.workers,
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", choices=paths.DATASETS, required=True)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--run", help=f"run of a network (default: seed{SEED})")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--limit", type=int, help="tune on the first N images only (with --dry-run, to time it)")
    parser.add_argument("--dry-run", action="store_true", help="write nothing; print the scores and projected time")
    parser.add_argument("--force", action="store_true", help="overwrite an existing frozen tuning")
    parser.add_argument("--configs-dir", type=Path, help="config folder (default: repo configs/)")
    args = parser.parse_args(argv)

    target = postproc_file(args.data, args.exp, args.configs_dir)
    if not args.dry_run:
        if target.exists() and not args.force:
            parser.error(f"{target} exists: a frozen tuning is not overwritten without --force")
        if args.limit:
            parser.error("--limit is for timing with --dry-run; the frozen tuning uses every image of val_tune_sub")
    cfg = load_config(args.data, args.exp, root=args.configs_dir, tuned=False)
    gt = coco_io.load_gt(args.data, SPLIT)
    total_images = len(gt["images"])
    if args.limit:
        gt = first_images(gt, args.limit)
    polygon, category = PolygonParams.from_config(cfg.postproc), gt["categories"][0]["id"]
    grid = tuning.load_grid(args.configs_dir)
    tic = time.perf_counter()
    if cfg.model.arch == "classical":
        job = infer.B0Job(args.data, SPLIT, B0Params.from_config(cfg.postproc.b0), polygon, category)
        report = tuning.tune_b0(job, gt, grid, args.workers)
    else:
        job = infer.NetworkJob(
            args.data,
            infer.network_source(cfg),
            args.run or paths.run_name(SEED),
            SPLIT,
            infer.InstanceParams.from_config(cfg.postproc),
            polygon,
            category,
        )
        report = tuning.tune_network(job, gt, grid, int(cfg.model.classes) == 1, args.workers)
    wall = time.perf_counter() - tic
    summary = {"chosen": report["chosen"], "score": report["score"], "seconds": report["seconds"], "wall": round(wall)}
    if args.limit:
        summary["projected_minutes_all_images"] = round(wall * total_images / len(gt["images"]) / 60, 1)
    print(json.dumps(summary))
    if args.dry_run:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Frozen post-processing of {args.exp} on {args.data} (docs/EXPERIMENTS.md §4), tuned on {SPLIT} and\n"
        "# written by scripts/tune_postproc.py; do not edit. load_config merges the postproc section only.\n"
    )
    document = frozen_document(args, report, tuning.grid_file(args.configs_dir))
    target.write_text(header + OmegaConf.to_yaml(OmegaConf.create(document)), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
