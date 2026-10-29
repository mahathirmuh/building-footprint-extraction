"""WHU (docs/DATA.md §3.2, option R): official 0.2 m tiles -> 512² quadrants with buildings -> Pix2Poly's 224 × 224
COCO patches, with two ground truths on the same patches.

    python scripts/prepare_whu_pix2poly.py                        # train, val and test
    python scripts/prepare_whu_pix2poly.py --split test           # one folder
    python scripts/prepare_whu_pix2poly.py --official-coco FILE   # touching pairs in an official COCO file
    python scripts/prepare_whu_pix2poly.py --cache train_dedup val_tune test  # instance maps of the main GT

Input under $DENSEPOLY_DATA/whu: raw_0.2/{train,val,test}/{image,label}/<stem>.tif (official 1024² tiles) and
official_coco_0.2/{train,validation,test}.json (official COCO).
Output per folder in $DENSEPOLY_DATA/whu/pix2poly_224/<folder>/: annotation.json (main ground truth, official
buildings), annotation_p2p.json (Pix2Poly's ground truth) and patches.csv (offset of every patch in its official
tile); the split lists whu_*.txt and whu_stats.json in splits/. The conversion logic is Pix2Poly's (MIT), see
src/densepoly/data/whu.py. --cache writes $DENSEPOLY_DATA/instances/whu/<split>/<image_id>.png from the main ground
truth, an index.csv per split, and splits/whu_raster_stats.json.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from densepoly import paths
from densepoly.data import whu
from densepoly.labels import instances
from densepoly.utils.io import read_json, write_json, write_lines
from densepoly.utils.provenance import git_revision, timestamp

FOLDERS = ("train", "val", "test")
SPLIT_OF_FOLDER = {"train": "train_dedup", "val": "val_tune", "test": "test"}
CACHE_SPLITS = tuple(SPLIT_OF_FOLDER.values())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", nargs="+", choices=FOLDERS, default=[], help="raw folders to convert (default: all)")
    parser.add_argument("--official-coco", type=Path, help="official WHU 0.2 m COCO file: count touching pairs")
    parser.add_argument("--cache", nargs="+", choices=CACHE_SPLITS, metavar="SPLIT", help="cache instance maps")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--splits-dir", type=Path, help="folder of split files and stats (default: repo splits/)")
    args = parser.parse_args(argv)
    folders = args.split or ([] if args.official_coco or args.cache else list(FOLDERS))

    out = args.splits_dir or paths.splits_dir()
    stats_file = out / "whu_stats.json"
    stats = read_json(stats_file) if stats_file.exists() else {}
    root = paths.dataset_dir("whu")
    for folder in folders:
        raw_dir = root / "raw_0.2" / folder
        coco_file = root / "official_coco_0.2" / f"{whu.OFFICIAL_COCO_NAMES[folder]}.json"
        for needed in (raw_dir / "label", raw_dir / "image", coco_file):
            if not needed.exists():
                raise SystemExit(f"missing {needed}; see docs/DATA.md §1 for the expected layout")
        result = whu.convert_split(raw_dir, folder, whu.official_by_tile(read_json(coco_file)), args.workers)
        split = SPLIT_OF_FOLDER[folder]
        write_json(paths.annotation_file("whu", split), whu.coco_dict(result.images, result.main), indent=None)
        write_json(paths.annotation_file("whu", split, "p2p"), whu.coco_dict(result.images, result.p2p), indent=None)
        rows = [",".join(str(p[k]) for k in whu.PATCH_INDEX_COLUMNS) for p in result.patches]
        write_lines(paths.patch_index("whu", split), [",".join(whu.PATCH_INDEX_COLUMNS), *rows])
        names = [img["file_name"] for img in result.images]
        counts = dict(result.counts)
        if folder == "val":
            sub, select = whu.split_val_by_tile(names)
            write_lines(paths.split_list("whu", "val_tune", out), names)
            write_lines(paths.split_list("whu", "val_tune_sub", out), sub)
            write_lines(paths.split_list("whu", "val_select", out), select)
            counts |= {"val_tune_sub": len(sub), "val_select": len(select)}
        else:
            write_lines(paths.split_list("whu", split, out), names)
        stats[folder] = {"created": timestamp(), "git": git_revision(), "official_coco": coco_file.name, **counts}
    if args.official_coco:
        anns = read_json(args.official_coco)["annotations"]
        stats["official_coco"] = {
            "created": timestamp(),
            "file": args.official_coco.name,
            "annotations": len(anns),
            "touching_pairs": whu.touching_pairs(anns),
        }
    write_json(stats_file, stats)
    print(json.dumps(stats, indent=1))
    if args.cache:
        raster_file = out / "whu_raster_stats.json"
        raster = read_json(raster_file) if raster_file.exists() else {}
        for split in args.cache:
            totals = instances.cache_instances("whu", split, out)
            raster[split] = {"created": timestamp(), "git": git_revision(), **totals}
        write_json(raster_file, raster)
        print(json.dumps(raster, indent=1))


if __name__ == "__main__":
    main()
