"""Check the CrowdAI download and cache instance maps (docs/DATA.md §1-2).

    python scripts/prepare_crowdai.py --stats                            # layout + annotation statistics
    python scripts/prepare_crowdai.py --cache train_dedup val_tune test  # uint16 instance maps (after dedup)

Expected layout under $DENSEPOLY_DATA/crowdai: {train,val}/images/*.jpg and {train,val}/annotation.json.
--stats writes splits/crowdai_stats.json; --cache writes $DENSEPOLY_DATA/instances/crowdai/<split>/<image_id>.png,
an index.csv per split, and splits/crowdai_raster_stats.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from densepoly import paths
from densepoly.data import crowdai
from densepoly.utils.io import read_json, read_lines, write_json, write_lines, write_png
from densepoly.utils.provenance import git_revision, timestamp

CACHE_SPLITS = ("train_dedup", "val_tune", "test")


def write_stats(root: Path, out: Path) -> dict:
    report = {"created": timestamp(), "git": git_revision()}
    for source in ("train", "val"):
        annotation = root / source / "annotation.json"
        if not annotation.is_file():
            raise SystemExit(f"missing {annotation}; see docs/DATA.md §1 for the expected layout")
        coco = crowdai.load_coco(annotation)
        report[source] = crowdai.annotation_stats(coco) | crowdai.check_files(coco, root / source / "images")
    write_json(out / "crowdai_stats.json", report)
    return report


def write_cache(root: Path, splits: list[str], out: Path) -> dict:
    stats_file = out / "crowdai_raster_stats.json"
    summary = read_json(stats_file) if stats_file.exists() else {}
    cocos = {}
    for split in splits:
        source = crowdai.source_folder(split)
        if source not in cocos:
            cocos[source] = crowdai.load_coco(root / source / "annotation.json")
        coco = cocos[source]
        by_name = {img["file_name"]: img for img in coco.dataset["images"]}
        totals = {"images": 0, "instances": 0, "skipped": 0, "empty": 0, "overlap_px": 0, "hidden": 0}
        index = ["file_name,image_id,instances"]
        for name in read_lines(paths.split_list("crowdai", split, out)):
            img = by_name[name]
            result = crowdai.rasterize_instances(coco.imgToAnns.get(img["id"], []), img["height"], img["width"])
            write_png(paths.instance_cache("crowdai", split, img["id"]), result.instance_map)
            index.append(f"{name},{img['id']},{len(result.annotation_ids)}")
            totals["images"] += 1
            totals["instances"] += len(result.annotation_ids)
            totals["skipped"] += result.n_skipped
            totals["empty"] += result.n_empty
            totals["overlap_px"] += result.overlap_px
            totals["hidden"] += result.n_hidden
        write_lines(paths.instance_dir("crowdai", split) / "index.csv", index)
        summary[split] = {"created": timestamp(), "git": git_revision(), **totals}
    write_json(stats_file, summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats", action="store_true", help="check the download, write splits/crowdai_stats.json")
    parser.add_argument("--cache", nargs="+", choices=CACHE_SPLITS, metavar="SPLIT", help="cache instance maps")
    parser.add_argument("--splits-dir", type=Path, help="folder of split files and stats (default: repo splits/)")
    args = parser.parse_args(argv)
    if not (args.stats or args.cache):
        parser.error("nothing to do: give --stats and/or --cache")

    root, out = paths.dataset_dir("crowdai"), args.splits_dir or paths.splits_dir()
    if args.stats:
        report = write_stats(root, out)
        summary = {s: {k: v for k, v in report[s].items() if k != "categories"} for s in ("train", "val")}
        print(json.dumps(summary, indent=1))
    if args.cache:
        print(json.dumps(write_cache(root, args.cache, out), indent=1))


if __name__ == "__main__":
    main()
