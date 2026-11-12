"""Boxes of the FULL+Y detector on a split (docs/METHOD.md §8), for the seeds of scripts/predict.py --exp full_y.

    python scripts/predict_yolo.py --data crowdai --split val_tune_sub
    python scripts/predict_yolo.py --data crowdai --split test

Writes $DENSEPOLY_OUT/yolo/<data>/boxes/<split>.json: {"boxes": {image id: [[x, y, confidence, w, h, angle], ...]}, ...}
with x, y the centre in pixels of the tile, every box of confidence >= --min-conf (below the smallest c_y of the grid),
and the seconds of the detector per image. Ultralytics (AGPL-3.0) is imported here, never by the densepoly package.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from densepoly import paths
from densepoly.data.dataset import tile_refs
from densepoly.utils.io import write_json
from densepoly.utils.provenance import git_revision, timestamp

IMAGE_SIZE = {"crowdai": 320, "whu": 224}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", choices=("crowdai",), required=True, help="WHU patches are not stored as images")
    parser.add_argument("--split", choices=paths.SPLITS, required=True)
    parser.add_argument("--name", default="seed0", help="run folder of scripts/train_yolo.py")
    parser.add_argument("--min-conf", type=float, default=0.05)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--splits-dir", type=Path, help="folder of the split lists (default: repo splits/)")
    args = parser.parse_args(argv)

    from ultralytics import YOLO

    weights = paths.yolo_dir(args.data) / "runs" / args.name / "weights" / "best.pt"
    model = YOLO(str(weights))
    refs = tile_refs(args.data, args.split, args.splits_dir)
    boxes, seconds = {}, 0.0
    for start in range(0, len(refs), args.batch):
        part = refs[start : start + args.batch]
        tic = time.perf_counter()
        results = model.predict(
            [str(ref.image) for ref in part], imgsz=IMAGE_SIZE[args.data], conf=args.min_conf, device=0, verbose=False
        )
        seconds += time.perf_counter() - tic
        for ref, result in zip(part, results, strict=True):
            rows = []
            if result.obb is not None and len(result.obb):
                for (x, y, w, h, angle), score in zip(result.obb.xywhr.tolist(), result.obb.conf.tolist(), strict=True):
                    rows.append([round(x, 2), round(y, 2), round(score, 4), round(w, 2), round(h, 2), round(angle, 4)])
            boxes[str(int(ref.key))] = rows
    record = {
        "dataset": args.data,
        "split": args.split,
        "created": timestamp(),
        "git": git_revision(),
        "weights": weights.relative_to(paths.out_root()).as_posix(),
        "min_conf": args.min_conf,
        "images": len(refs),
        "n_boxes": sum(len(rows) for rows in boxes.values()),
        "seconds_per_image": seconds / max(len(refs), 1),
        "boxes": boxes,
    }
    write_json(paths.yolo_boxes_file(args.data, args.split), record, indent=None)
    print(json.dumps({key: record[key] for key in ("images", "n_boxes", "seconds_per_image")}))


if __name__ == "__main__":
    main()
