"""Cost of a trained run (docs/EXPERIMENTS.md §6; densepoly.eval.efficiency).

    python scripts/benchmark.py --data crowdai --exp full_p0 --seed 0

Measures the number of parameters, the speed of the network (batch 1, padded tile size, AMP, 20 warmup passes and 200
timed ones) and the post-processing per image on one CPU thread, on --images random tiles of --split with the frozen
post-processing. Writes outputs/metrics/<data>/<exp>/<run>/efficiency.json, which scripts/tables.py reads. The record
keeps the GPU load seen before the measurement: on a shared GPU, repeat it when another job was computing.
"""

from __future__ import annotations

import os

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # before numpy and torch are loaded
    os.environ[variable] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from densepoly import infer, paths  # noqa: E402
from densepoly.config import load_config, postproc_file  # noqa: E402
from densepoly.data.dataset import TileDataset  # noqa: E402
from densepoly.eval import coco_io, efficiency  # noqa: E402
from densepoly.models.factory import input_channels  # noqa: E402
from densepoly.postproc.polygonize import PolygonParams  # noqa: E402
from densepoly.utils.io import write_json  # noqa: E402
from densepoly.utils.provenance import git_revision, timestamp  # noqa: E402


def gpu_load() -> dict | None:
    """Utilization (%) and memory used (MiB) of the first GPU as nvidia-smi sees them now; None without nvidia-smi."""
    query = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"]
    try:
        first = subprocess.run(query, capture_output=True, text=True, check=True).stdout.splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None
    utilization, used = (int(value) for value in first.split(","))
    return {"utilization_percent": utilization, "memory_used_mib": used}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", choices=paths.DATASETS, required=True)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=paths.SPLITS, default="test", help="tiles of the post-processing timing")
    parser.add_argument("--images", type=int, default=200, help="tiles of the post-processing timing")
    parser.add_argument("--configs-dir", type=Path, help="config folder (default: repo configs/)")
    parser.add_argument("--splits-dir", type=Path, help="folder of the split lists (default: repo splits/)")
    parser.add_argument("--loader-workers", type=int, default=2, help="processes reading the tiles")
    parser.add_argument("overrides", nargs="*", help="config overrides key=value (not of the post-processing)")
    args = parser.parse_args(argv)

    if not postproc_file(args.data, args.exp, args.configs_dir).exists():
        parser.error("the post-processing of this experiment is not frozen yet (scripts/tune_postproc.py)")
    if any(item.startswith("postproc.") for item in args.overrides):
        parser.error("the post-processing is the frozen tuning; no postproc override")
    cfg = load_config(args.data, args.exp, args.overrides, root=args.configs_dir)
    run, source = paths.run_name(args.seed), infer.network_source(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    before = gpu_load()
    model, _ = infer.load_network(cfg, paths.checkpoint_file(args.data, source, run), device)
    side = int(cfg.data.pad_to)
    speed = efficiency.network_speed(model, (1, input_channels(cfg), side, side), device, amp=bool(cfg.train.amp))

    dataset = TileDataset(cfg, args.split, "predict", splits_dir=args.splits_dir)
    picked = sorted(np.random.default_rng(0).permutation(len(dataset))[: args.images].tolist())
    loader = DataLoader(Subset(dataset, picked), batch_size=8, num_workers=args.loader_workers)
    category = coco_io.load_gt(args.data, args.split)["categories"][0]["id"]
    job = infer.NetworkJob(
        args.data,
        source,
        run,
        args.split,
        infer.InstanceParams.from_config(cfg.postproc),
        PolygonParams.from_config(cfg.postproc),
        category,
    )
    channels = [(image_id, tile) for image_id, tile, _ in infer.network_channels(model, loader, device, int(cfg.data.tile_size))]
    postproc = {**efficiency.postproc_time(job, channels), "split": args.split, "threads": 1}
    record = {
        "dataset": args.data,
        "exp": args.exp,
        "run": run,
        "created": timestamp(),
        "git": git_revision(),
        "checkpoint": f"{source}/{run}/best.pt",
        "parameters": efficiency.count_parameters(model),
        **speed,
        "postproc": postproc,
        "postproc_config": {"instance": str(cfg.postproc.instance), "polygon": str(cfg.postproc.polygon)},
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "none (CPU)",
        "cpu": efficiency.cpu_name(),
        "gpu_load_before": before,
        "gpu_load_after": gpu_load(),
    }
    write_json(paths.efficiency_file(args.data, args.exp, run), record)
    print(json.dumps(record))


if __name__ == "__main__":
    main()
