"""Evaluate the predictions of one run on one split (docs/EVALUATION.md; densepoly.eval.run_eval).

    python scripts/evaluate.py --data crowdai --exp full_p2 --seed 0 --split test
    python scripts/evaluate.py --data whu --exp full_p0 --seed 0 --split test --variant p2p   # Pix2Poly's GT
    python scripts/evaluate.py --data crowdai --exp hisup_release --split val_clean --released

Predictions are read from outputs/preds/<data>/<exp>/<run>/<split>.json (run seed<k>, or ref without --seed) unless
--pred names another file; a <split>.meta.json next to them (checkpoint, postproc, efficiency) is copied into the
record. --released checks predictions leniently, as released by another paper. Writes
outputs/metrics/<data>/<exp>/<run>/<split>[_<variant>].json with its per-image and per-pair tables and the cached
COCOeval results (not for val_full, where they would be large and no bootstrap is planned).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from densepoly import paths
from densepoly.eval import run_eval
from densepoly.utils.io import read_json

SHOWN = ("AP", "AP_boundary", "IoU", "CIoU", "CIoU_p2p", "PoLiS", "MTA", "PoLiS_paired", "MTA_inst", "merge_rate")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", choices=paths.DATASETS, required=True)
    parser.add_argument("--exp", required=True, help="experiment code, e.g. full_p2, b0, hisup_release")
    parser.add_argument("--seed", type=int, help="seed of a trained run; omitted for seedless runs (ref)")
    parser.add_argument("--split", choices=paths.SPLITS, required=True)
    parser.add_argument("--variant", help="ground truth variant (WHU: p2p)")
    parser.add_argument("--pred", type=Path, help="prediction file (default: outputs/preds/...)")
    parser.add_argument("--released", action="store_true", help="lenient checks for released predictions")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--splits-dir", type=Path, help="folder of the density files (default: repo splits/)")
    args = parser.parse_args(argv)

    run = paths.run_name(args.seed)
    pred_file = args.pred or paths.pred_file(args.data, args.exp, run, args.split)
    meta_file = pred_file.with_name(f"{pred_file.stem}.meta.json")
    record = run_eval.evaluate(
        args.data,
        args.split,
        args.exp,
        run,
        read_json(pred_file),
        variant=args.variant,
        released=args.released,
        workers=args.workers,
        meta=read_json(meta_file) if meta_file.exists() else None,
        pred_file=pred_file,
        cache_evalimgs=args.split != "val_full",
        splits_dir=args.splits_dir,
    )
    print(json.dumps({key: record["overall"][key] for key in SHOWN}))
    print(json.dumps({"pred_check": record["pred_check"], "seconds": record["seconds"]}))


if __name__ == "__main__":
    main()
