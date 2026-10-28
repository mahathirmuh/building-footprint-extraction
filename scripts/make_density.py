"""Density strata of a split (docs/DATA.md §4). Run before any model is evaluated; the output is then frozen and
an existing file is never overwritten without --force.

    python scripts/make_density.py --data crowdai --split test       # thresholds come from this split
    python scripts/make_density.py --data crowdai --split val_tune   # reuses the test thresholds
    python scripts/make_density.py --data crowdai --split val_clean  # likewise (HiSup comparison per stratum)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from densepoly import paths
from densepoly.data import crowdai, density
from densepoly.utils.io import read_json, read_lines, write_json
from densepoly.utils.provenance import git_revision, timestamp


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", choices=["crowdai", "whu"], required=True)
    parser.add_argument("--split", choices=["test", "val_tune", "val_clean"], required=True)
    parser.add_argument("--splits-dir", type=Path, help="folder of split files (default: repo splits/)")
    parser.add_argument("--force", action="store_true", help="overwrite; only allowed before any model evaluation")
    args = parser.parse_args(argv)

    out = args.splits_dir or paths.splits_dir()
    target = paths.density_file(args.data, args.split, out)
    if target.exists() and not args.force:
        raise SystemExit(f"{target} already exists and is frozen (docs/DATA.md §4)")
    thresholds = None
    if args.split != "test":
        test_file = paths.density_file(args.data, "test", out)
        if not test_file.exists():
            raise SystemExit(f"create the test strata first: thresholds come from {test_file.name}")
        thresholds = read_json(test_file)["thresholds"]

    radius = int(OmegaConf.load(paths.configs_dir() / "data" / f"{args.data}.yaml").labels.contact_radius_px)
    coco = crowdai.load_coco(paths.annotation_file(args.data, args.split))
    by_name = {img["file_name"]: img for img in coco.dataset["images"]}
    counts, ratios = {}, {}
    for name in read_lines(paths.split_list(args.data, args.split, out)):
        img = by_name[name]
        result = crowdai.rasterize_instances(coco.imgToAnns.get(img["id"], []), img["height"], img["width"])
        counts[img["id"]] = len(result.annotation_ids)
        ratios[img["id"]] = density.touching_ratio(result.instance_map, radius)

    record = density.strata_record(counts, ratios, thresholds)
    header = {
        "dataset": args.data,
        "split": args.split,
        "created": timestamp(),
        "git": git_revision(),
        "thresholds_from": f"{args.data}/test",
        "contact_radius_px": radius,
    }
    write_json(target, header | record)
    print(json.dumps({"thresholds": record["thresholds"], **{k: record[k]["sizes"] for k in ("main", "secondary")}}))


if __name__ == "__main__":
    main()
