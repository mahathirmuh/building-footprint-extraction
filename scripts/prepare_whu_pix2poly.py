"""WHU protocol A (docs/DATA.md §3.2): Pix2Poly's conversion of the 0.2 m rasters into 224 × 224 COCO patches.

    python scripts/prepare_whu_pix2poly.py                        # train, val and test
    python scripts/prepare_whu_pix2poly.py --split test           # one folder
    python scripts/prepare_whu_pix2poly.py --official-coco FILE   # touching pairs in the official 0.2 m COCO file

Input: $DENSEPOLY_DATA/whu/raw_0.2/{train,val,test}/{image,label}/*.tif (folder names as in Pix2Poly).
Output: $DENSEPOLY_DATA/whu/pix2poly_224/{train,val,test}/{images/, annotation.json}, plus the split lists whu_*.txt
and whu_stats.json in splits/. The conversion logic is Pix2Poly's (MIT), see src/densepoly/data/whu.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from densepoly import paths
from densepoly.data import whu
from densepoly.utils.io import read_json, write_json, write_lines
from densepoly.utils.provenance import git_revision, timestamp

FOLDERS = ("train", "val", "test")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", nargs="+", choices=FOLDERS, default=[], help="raw folders to convert (default: all)")
    parser.add_argument("--official-coco", type=Path, help="official WHU 0.2 m COCO file: count touching pairs")
    parser.add_argument("--no-images", action="store_true", help="write annotation.json only, not the patch images")
    parser.add_argument("--splits-dir", type=Path, help="folder of split files and stats (default: repo splits/)")
    args = parser.parse_args(argv)
    folders = args.split or ([] if args.official_coco else list(FOLDERS))

    out = args.splits_dir or paths.splits_dir()
    stats_file = out / "whu_stats.json"
    stats = read_json(stats_file) if stats_file.exists() else {}
    raw, converted = paths.dataset_dir("whu") / "raw_0.2", paths.dataset_dir("whu") / "pix2poly_224"
    for folder in folders:
        if not (raw / folder / "label").is_dir():
            raise SystemExit(f"missing {raw / folder}/label; see docs/DATA.md §1 for the expected layout")
        counts, names = whu.convert_split(raw / folder, converted / folder, folder, save_images=not args.no_images)
        if folder == "val":
            sub, select = whu.split_val_by_tile(names)
            write_lines(paths.split_list("whu", "val_tune", out), names)
            write_lines(paths.split_list("whu", "val_tune_sub", out), sub)
            write_lines(paths.split_list("whu", "val_select", out), select)
            counts |= {"val_tune_sub": len(sub), "val_select": len(select)}
        else:
            write_lines(paths.split_list("whu", "train_dedup" if folder == "train" else "test", out), names)
        stats[folder] = {"created": timestamp(), "git": git_revision(), **counts}
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


if __name__ == "__main__":
    main()
