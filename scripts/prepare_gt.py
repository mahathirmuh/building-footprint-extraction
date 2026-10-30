"""Ground truth of the evaluated splits (docs/EVALUATION.md §1): outputs/gt/<dataset>/<split>[_<variant>].json.

    python scripts/prepare_gt.py --data crowdai            # every evaluated split (EVAL_SPLITS), val_full included
    python scripts/prepare_gt.py --data whu --split test   # the main and the p2p ground truth of the WHU test

Every file keeps all images of its split; its "info" records the source file, the annotation id offset, the dropped
WHU negative samples and the special cases kept as released (densepoly.eval.coco_io). Each source annotation file is
read once for all the splits it holds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from densepoly import paths
from densepoly.eval import coco_io
from densepoly.utils.io import read_json

EVAL_SPLITS = {
    "crowdai": ("val_tune", "val_tune_sub", "val_select", "test", "val_clean", "val_full"),
    "whu": ("val_tune", "val_tune_sub", "val_select", "test"),
}
SHOWN = (
    "images",
    "images_without_annotations",
    "annotations",
    "dropped_negative_samples",
    "ann_id_offset",
    "empty_masks",
    "area_field_zero",
    "multi_ring_annotations",
    "ring_convention",
)


def prepare(dataset: str, splits: list[str], splits_dir: Path | None) -> list[dict]:
    """Write the ground truth of every split and variant; return their "info" in writing order."""
    infos = []
    for variant in (None, *paths.GT_VARIANTS.get(dataset, ())):
        by_source: dict[Path, list[str]] = {}
        for split in splits:
            by_source.setdefault(paths.annotation_file(dataset, split, variant), []).append(split)
        for source, group in by_source.items():
            coco_data = read_json(source)
            infos.extend(coco_io.prepare_gt(dataset, split, variant, splits_dir, coco_data) for split in group)
    return infos


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", choices=sorted(EVAL_SPLITS), required=True)
    parser.add_argument("--split", nargs="+", help="splits to write (default: every evaluated split)")
    parser.add_argument("--splits-dir", type=Path, help="folder of split files (default: repo splits/)")
    args = parser.parse_args(argv)
    splits = args.split or list(EVAL_SPLITS[args.data])
    unknown = sorted(set(splits) - set(EVAL_SPLITS[args.data]))
    if unknown:
        parser.error(f"--split for {args.data} must be among {EVAL_SPLITS[args.data]}, not {unknown}")
    for info in prepare(args.data, splits, args.splits_dir):
        print(json.dumps({key: info[key] for key in ("split", "variant", *SHOWN)}))


if __name__ == "__main__":
    main()
