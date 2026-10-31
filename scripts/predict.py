"""Predictions of one experiment on one split (docs/EVALUATION.md §1; densepoly.infer).

    python scripts/predict.py --data crowdai --exp b0 --split test --workers 12 --time-sample 50
    python scripts/predict.py --data crowdai --exp b4 --seed 0 --split val_tune_sub --save-channels
    python scripts/predict.py --data crowdai --exp b5 --seed 0 --split test --from-channels --workers 12

Writes outputs/preds/<data>/<exp>/<run>/<split>.json and <split>.meta.json (post-processing, timings, efficiency),
which scripts/evaluate.py copies into the metrics. The test split needs the frozen tuning
configs/postproc/<data>/<exp>.yaml and takes no post-processing override: parameters are never tuned on test
(docs/EXPERIMENTS.md §4). B5 and B5′ use the checkpoint of their source experiment.

--save-channels keeps the network output per image in outputs/channels/, for scripts/tune_postproc.py and for other
post-processing of the same network; --from-channels reads it back instead of running the network. --time-sample N
times N random images of B0 in one process from the RGB tile, MBI included.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from densepoly import infer, paths
from densepoly.classical.b0 import B0Params
from densepoly.config import load_config, postproc_file
from densepoly.data.dataset import TileDataset
from densepoly.eval import coco_io
from densepoly.postproc.polygonize import PolygonParams
from densepoly.train import worker_setup
from densepoly.utils.io import write_json
from densepoly.utils.provenance import git_revision, timestamp


def predict_classical(args, cfg, gt: dict) -> tuple[list[dict], list[dict], dict]:
    job = infer.B0Job(
        args.data,
        args.split,
        B0Params.from_config(cfg.postproc.b0),
        PolygonParams.from_config(cfg.postproc),
        gt["categories"][0]["id"],
    )
    results, timings = infer.predict_b0(job, [image["id"] for image in gt["images"]], args.workers)
    efficiency = None
    if args.time_sample:
        efficiency = infer.b0_seconds_from_rgb(job, gt["images"], cfg.priors.mbi_scales_px, args.time_sample)
    return results, timings, {"checkpoint": None, "efficiency": efficiency}


def predict_with_network(args, cfg, gt: dict, run: str) -> tuple[list[dict], list[dict], dict]:
    source = infer.network_source(cfg)
    job = infer.NetworkJob(
        args.data,
        source,
        run,
        args.split,
        infer.InstanceParams.from_config(cfg.postproc),
        PolygonParams.from_config(cfg.postproc),
        gt["categories"][0]["id"],
    )
    checkpoint = paths.checkpoint_file(args.data, source, run)
    if args.from_channels:
        results, timings = infer.predict_saved(job, [image["id"] for image in gt["images"]], args.workers)
        return results, timings, {"checkpoint": None, "channels_of": f"{source}/{run}", "efficiency": None}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, state = infer.load_network(cfg, checkpoint, device)
    dataset = TileDataset(cfg, args.split, "predict", splits_dir=args.splits_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.loader_workers,
        worker_init_fn=worker_setup if args.loader_workers else None,
    )
    results, timings = infer.predict_network(job, model, loader, device, int(cfg.data.tile_size), args.save_channels)
    meta = {"checkpoint": f"{source}/{run}/best.pt", "selected": state, "device": str(device), "efficiency": None}
    return results, timings, meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", choices=paths.DATASETS, required=True)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--split", choices=paths.SPLITS, required=True)
    parser.add_argument("--seed", type=int, help="seed of a trained run; omitted for B0 (run ref)")
    parser.add_argument("--run", help="named run, e.g. pilot_r5 (instead of --seed)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--batch-size", type=int, default=8, help="tiles per forward pass of a network")
    parser.add_argument("--loader-workers", type=int, default=2, help="processes reading tiles for a network")
    parser.add_argument("--save-channels", action="store_true", help="keep the network output per image")
    parser.add_argument("--from-channels", action="store_true", help="read saved channels, run no network")
    parser.add_argument("--time-sample", type=int, default=0, metavar="N", help="B0: time N images from RGB")
    parser.add_argument("--configs-dir", type=Path, help="config folder (default: repo configs/)")
    parser.add_argument("--splits-dir", type=Path, help="folder of the split lists (default: repo splits/)")
    parser.add_argument("overrides", nargs="*", help="config overrides key=value")
    args = parser.parse_args(argv)

    tuned = postproc_file(args.data, args.exp, args.configs_dir)
    if args.split == "test":
        if not tuned.exists():
            parser.error(f"{tuned} does not exist: tune on val_tune_sub and freeze it before predicting on test")
        if any(item.startswith("postproc.") for item in args.overrides):
            parser.error("the post-processing of test is the frozen tuning; no postproc override")
    cfg = load_config(args.data, args.exp, args.overrides, root=args.configs_dir)
    run = args.run or paths.run_name(args.seed)
    gt = coco_io.load_gt(args.data, args.split)
    tic = time.perf_counter()
    if cfg.model.arch == "classical":
        results, timings, source = predict_classical(args, cfg, gt)
    else:
        if run == "ref":
            parser.error("a network needs --seed or --run")
        results, timings, source = predict_with_network(args, cfg, gt, run)
    wall = time.perf_counter() - tic
    results.sort(key=lambda result: result["image_id"])  # stable: the order of the instances inside an image stays
    coco_io.check_predictions(results, gt, strict=True)  # our outputs must follow docs/EVALUATION.md §1 exactly

    pred_file = paths.pred_file(args.data, args.exp, run, args.split)
    seconds = {"wall": round(wall, 1), "workers": args.workers}
    for stage in ("network", "instances", "polygons"):
        if stage in timings[0]:
            seconds[f"{stage}_mean"] = float(np.mean([t[stage] for t in timings]))
    meta = {
        "dataset": args.data,
        "exp": args.exp,
        "run": run,
        "split": args.split,
        "created": timestamp(),
        "git": git_revision(),
        **source,
        "postproc": OmegaConf.to_container(cfg.postproc),
        "tuned_file": tuned.relative_to(tuned.parents[3]).as_posix() if tuned.exists() else None,
        "images": len(gt["images"]),
        "predictions": len(results),
        "fallbacks": sum(r["fallback"] for r in results),
        "seconds": seconds,
    }
    write_json(pred_file, results, indent=None)
    write_json(pred_file.with_name(f"{pred_file.stem}.meta.json"), meta)
    print(json.dumps({k: meta[k] for k in ("images", "predictions", "fallbacks", "seconds", "efficiency")}))


if __name__ == "__main__":
    main()
