"""Predictions of one experiment on one split (docs/EVALUATION.md §1; densepoly.infer).

    python scripts/predict.py --data crowdai --exp b0 --split val_tune_sub --workers 12
    python scripts/predict.py --data crowdai --exp b0 --split test --workers 12 --time-sample 50

Writes outputs/preds/<data>/<exp>/<run>/<split>.json and <split>.meta.json (post-processing, timings, efficiency),
which scripts/evaluate.py copies into the metrics. The test split needs the frozen tuning
configs/postproc/<data>/<exp>.yaml and takes no post-processing override: parameters are never tuned on test
(docs/EXPERIMENTS.md §4). --time-sample N times N random images in one process from the RGB tile, MBI included.
Phase 4 runs the classical B0 only.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from densepoly import infer, paths
from densepoly.classical.b0 import B0Params
from densepoly.config import load_config, postproc_file
from densepoly.eval import coco_io
from densepoly.postproc.polygonize import PolygonParams
from densepoly.utils.io import write_json
from densepoly.utils.provenance import git_revision, timestamp


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", choices=paths.DATASETS, required=True)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--split", choices=paths.SPLITS, required=True)
    parser.add_argument("--seed", type=int, help="seed of a trained run; omitted for B0 (run ref)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--time-sample", type=int, default=0, metavar="N", help="time N images from RGB")
    parser.add_argument("--configs-dir", type=Path, help="config folder (default: repo configs/)")
    parser.add_argument("overrides", nargs="*", help="config overrides key=value")
    args = parser.parse_args(argv)

    tuned = postproc_file(args.data, args.exp, args.configs_dir)
    if args.split == "test":
        if not tuned.exists():
            parser.error(f"{tuned} does not exist: tune on val_tune_sub and freeze it before predicting on test")
        if any(item.startswith("postproc.") for item in args.overrides):
            parser.error("the post-processing of test is the frozen tuning; no postproc override")
    cfg = load_config(args.data, args.exp, args.overrides, root=args.configs_dir)
    if cfg.model.arch != "classical":
        parser.error("network inference comes with the first model (Phase 5)")
    gt = coco_io.load_gt(args.data, args.split)
    job = infer.B0Job(
        args.data,
        args.split,
        B0Params.from_config(cfg.postproc.b0),
        PolygonParams.from_config(cfg.postproc),
        gt["categories"][0]["id"],
    )
    tic = time.perf_counter()
    results, timings = infer.predict_b0(job, [image["id"] for image in gt["images"]], args.workers)
    wall = time.perf_counter() - tic
    coco_io.check_predictions(results, gt, strict=True)  # our outputs must follow docs/EVALUATION.md §1 exactly
    efficiency = None
    if args.time_sample:
        efficiency = infer.b0_seconds_from_rgb(job, gt["images"], cfg.priors.mbi_scales_px, args.time_sample)

    run = paths.run_name(args.seed)
    pred_file = paths.pred_file(args.data, args.exp, run, args.split)
    meta = {
        "dataset": args.data,
        "exp": args.exp,
        "run": run,
        "split": args.split,
        "created": timestamp(),
        "git": git_revision(),
        "checkpoint": None,
        "postproc": OmegaConf.to_container(cfg.postproc),
        "tuned_file": tuned.relative_to(tuned.parents[3]).as_posix() if tuned.exists() else None,
        "images": len(gt["images"]),
        "predictions": len(results),
        "fallbacks": sum(r["fallback"] for r in results),
        "seconds": {
            "wall": round(wall, 1),
            "workers": args.workers,
            "instances_mean": float(np.mean([t["instances"] for t in timings])),
            "polygons_mean": float(np.mean([t["polygons"] for t in timings])),
        },
        "efficiency": efficiency,
    }
    write_json(pred_file, results, indent=None)
    write_json(pred_file.with_name(f"{pred_file.stem}.meta.json"), meta)
    print(json.dumps({k: meta[k] for k in ("images", "predictions", "fallbacks", "seconds", "efficiency")}))


if __name__ == "__main__":
    main()
