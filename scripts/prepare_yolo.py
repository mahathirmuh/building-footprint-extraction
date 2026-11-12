"""Export the training set of the FULL+Y detector (docs/METHOD.md §8, docs/EXPERIMENTS.md §7; densepoly.postproc.yolo_seeds).

    python scripts/prepare_yolo.py --data crowdai                 # train_dedup -> train, val_select -> val
    python scripts/prepare_yolo.py --data crowdai --limit 500     # the first 500 tiles of each split, to time it

One oriented box per building, from the instance maps alone (cv2.minAreaRect): still no polygon annotation. Writes
$DENSEPOLY_OUT/yolo/<data>/dataset/ in the Ultralytics layout: images/{train,val}/<id>.<ext> (links to the tiles),
labels/{train,val}/<id>.txt ("0 x1 y1 x2 y2 x3 y3 x4 y4", normalised) and data.yaml. This script does not import
Ultralytics (AGPL-3.0); scripts/train_yolo.py and scripts/predict_yolo.py do, in their own environment.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path

from densepoly import paths
from densepoly.data.dataset import tile_refs
from densepoly.postproc.yolo_seeds import obb_label_lines
from densepoly.utils.io import read_png, write_json, write_lines
from densepoly.utils.provenance import git_revision, timestamp

PARTS = {"train": "train_dedup", "val": "val_select"}


def export_tile(args) -> int:
    """Write the label file and the image link of one tile; returns its number of boxes."""
    dataset, split, key, image, root, part = args
    lines = obb_label_lines(read_png(paths.instance_cache(dataset, split, key)))
    write_lines(Path(root) / "labels" / part / f"{key}.txt", lines)
    link = Path(root) / "images" / part / f"{key}{Path(image).suffix}"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(image)
    return len(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", choices=("crowdai",), required=True, help="WHU patches are not stored as images")
    parser.add_argument("--limit", type=int, help="only the first N tiles of each split")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--splits-dir", type=Path, help="folder of the split lists (default: repo splits/)")
    args = parser.parse_args(argv)

    root = paths.yolo_dir(args.data) / "dataset"
    report = {"dataset": args.data, "created": timestamp(), "git": git_revision(), "limit": args.limit, "parts": {}}
    for part, split in PARTS.items():
        refs = tile_refs(args.data, split, args.splits_dir)[: args.limit]
        for folder in ("images", "labels"):
            (root / folder / part).mkdir(parents=True, exist_ok=True)
        jobs = [(args.data, split, ref.key, str(ref.image), str(root), part) for ref in refs]
        tic = time.perf_counter()
        if args.workers > 1:
            with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
                counts = pool.map(export_tile, jobs, chunksize=64)
        else:
            counts = [export_tile(job) for job in jobs]
        report["parts"][part] = {
            "split": split,
            "tiles": len(refs),
            "boxes": int(sum(counts)),
            "tiles_without_box": int(sum(count == 0 for count in counts)),
            "seconds": round(time.perf_counter() - tic, 1),
        }
    data_yaml = [f"path: {root}", "train: images/train", "val: images/val", "names:", "  0: building"]
    write_lines(root / "data.yaml", data_yaml)
    write_json(root / "export.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
