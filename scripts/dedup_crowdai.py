"""Deduplicate CrowdAI and write the train_dedup / val_tune / test splits (docs/DATA.md §2.2).

    python scripts/dedup_crowdai.py --data crowdai --benchmark 1000   # time 1,000 images first; writes nothing
    python scripts/dedup_crowdai.py --data crowdai                    # full run (ask before running it)

Hashes are cached per chunk under $DENSEPOLY_DATA/crowdai/hashes/, so an interrupted run resumes. Outputs: split
lists, crowdai_train_clusters.json and dedup_report.json in splits/, and the QA mosaic in
$DENSEPOLY_OUT/figures/qa/dedup_clusters.png.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from densepoly import paths
from densepoly.data import dedup
from densepoly.data.crowdai import list_images
from densepoly.utils.io import write_json, write_lines
from densepoly.utils.provenance import git_revision, timestamp

SPLITS = ("train_dedup", "val_tune", "val_tune_sub", "val_select", "test")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", choices=["crowdai"], required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--benchmark", type=int, metavar="N", help="hash N train images on one core, print timing")
    parser.add_argument("--splits-dir", type=Path, help="output folder for split files (default: repo splits/)")
    args = parser.parse_args(argv)

    root = paths.dataset_dir(args.data)
    image_dirs = {source: root / source / "images" for source in ("train", "val")}
    names = {source: list_images(folder) for source, folder in image_dirs.items()}
    if args.benchmark:
        files = [image_dirs["train"] / n for n in names["train"][: args.benchmark]]
        result = dedup.benchmark(files, len(names["train"]) + len(names["val"]), args.workers)
        print(json.dumps(result, indent=1))
        return

    hashes = {}
    for source in ("train", "val"):
        cache = root / "hashes" / f"{source}_aug_chunks"
        hashes[source] = dedup.hash_images([image_dirs[source] / n for n in names[source]], cache, args.workers)
        merged = [{"image_name": n, "hashes": list(h)} for n, h in zip(names[source], hashes[source], strict=True)]
        write_json(root / "hashes" / f"{source}_aug.json", merged, indent=None)

    result = dedup.build_splits(names["train"], hashes["train"], names["val"], hashes["val"])
    out = args.splits_dir or paths.splits_dir()
    for split in SPLITS:
        write_lines(paths.split_list(args.data, split, out), result.splits[split])
    write_json(out / f"{args.data}_train_clusters.json", result.train_clusters, indent=None)
    write_json(out / "dedup_report.json", {"created": timestamp(), "git": git_revision(), **result.report})

    qa_file = paths.out_root() / "figures" / "qa" / "dedup_clusters.png"
    qa_file.parent.mkdir(parents=True, exist_ok=True)
    dedup.cluster_mosaic(result.train_clusters, image_dirs["train"]).save(qa_file)
    print(json.dumps({key: result.report[key] for key in ("train", "val", "splits")}, indent=1))


if __name__ == "__main__":
    main()
