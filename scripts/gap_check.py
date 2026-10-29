"""Gap-map check on val_tune (PLAN.md Phase 2, docs/METHOD.md §2.3, §10): how well each gap mode and SE length L_g
separates narrow gaps between two buildings from the rest, to back the visual choice with numbers.

    python scripts/gap_check.py --data crowdai              # L_g in {7, 9, 11}
    python scripts/gap_check.py --data whu --tiles 1000     # L_g in {11, 13, 17}, 1,000 random patches

Narrow-gap pixels are background pixels that see two different buildings within r + 1 px (r = contact radius).
For every mode and L_g: AUC of gap pixels against the other background pixels and against roof pixels, i.e. the
probability that a random gap pixel responds more strongly (0.5 = no separation). Output:
$DENSEPOLY_OUT/metrics/<dataset>/phase2/gap_check_val_tune.json.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import disk

from densepoly import paths
from densepoly.config import load_config
from densepoly.labels.instances import load_instances
from densepoly.priors import cache
from densepoly.priors.gap import GAP_MODES, gap_map
from densepoly.priors.mbi import brightness
from densepoly.utils.io import write_json
from densepoly.utils.provenance import git_revision, timestamp

LENGTHS = {"crowdai": (7, 9, 11), "whu": (11, 13, 17)}
BINS = 4096


def narrow_gaps(instance_map: np.ndarray, radius: int) -> np.ndarray:
    """Background pixels with at least two different buildings inside a disk of the given radius."""
    footprint = disk(radius)
    big = np.iinfo(instance_map.dtype).max
    largest = ndi.maximum_filter(instance_map, footprint=footprint)
    smallest = ndi.minimum_filter(np.where(instance_map > 0, instance_map, big), footprint=footprint)
    return (instance_map == 0) & (largest > 0) & (smallest < big) & (largest != smallest)


def auc(positive_hist: np.ndarray, negative_hist: np.ndarray) -> float:
    """P(positive > negative) + P(equal) / 2 from two histograms on the same bins."""
    below = np.cumsum(negative_hist) - negative_hist  # negatives in lower bins
    wins = (positive_hist * below).sum() + 0.5 * (positive_hist * negative_hist).sum()
    return float(wins / max(positive_hist.sum() * negative_hist.sum(), 1))


def tiles(dataset: str, n: int | None, seed: int = 0):
    """(RGB, instance map) of val_tune images; WHU patches are cut from their official tiles."""
    index = paths.instance_dir(dataset, "val_tune") / "index.csv"
    with index.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if n is not None and n < len(rows):
        rows = [rows[i] for i in sorted(np.random.default_rng(seed).permutation(len(rows))[:n])]
    if dataset == "crowdai":
        folder = paths.image_dir(dataset, "val_tune")
        for row in rows:
            yield cache.read_rgb(folder / row["file_name"]), load_instances(dataset, "val_tune", row["image_id"])
        return
    with paths.patch_index(dataset, "val_tune").open(encoding="utf-8") as f:
        offsets = {r["image_id"]: r for r in csv.DictReader(f)}
    folder = paths.dataset_dir(dataset) / "raw_0.2" / paths.source_folder(dataset, "val_tune") / "image"
    current, rgb = None, None
    for row in sorted(rows, key=lambda r: offsets[r["image_id"]]["tile"]):
        patch = offsets[row["image_id"]]
        if patch["tile"] != current:
            current, rgb = patch["tile"], cache.read_rgb(folder / f"{patch['tile']}.tif")
        y, x = int(patch["y"]), int(patch["x"])
        yield rgb[y : y + 224, x : x + 224], load_instances(dataset, "val_tune", row["image_id"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", choices=paths.DATASETS, required=True)
    parser.add_argument("--tiles", type=int, help="random val_tune images to use (default: all)")
    args = parser.parse_args(argv)

    radius = int(load_config(args.data, "full_p2").labels.contact_radius_px) + 1
    settings = list(product(GAP_MODES, LENGTHS[args.data]))
    hists = {s: {k: np.zeros(BINS) for k in ("gap", "background", "roof")} for s in settings}
    counts = {"tiles": 0, "gap_px": 0}
    for rgb, instance_map in tiles(args.data, args.tiles):
        groups = {"gap": narrow_gaps(instance_map, radius)}
        groups["background"] = (instance_map == 0) & ~groups["gap"]
        groups["roof"] = instance_map > 0
        counts["tiles"] += 1
        counts["gap_px"] += int(groups["gap"].sum())
        b = brightness(rgb)
        for mode, length in settings:
            response = np.clip(gap_map(b, length, mode), 0.0, 1.0)  # brightness differences lie in [0, 1]
            for key, mask in groups.items():
                hists[mode, length][key] += np.histogram(response[mask], bins=BINS, range=(0.0, 1.0))[0]

    results = []
    for mode, length in settings:
        h = hists[mode, length]
        results.append(
            {
                "mode": mode,
                "gap_se_length_px": length,
                "auc_gap_vs_background": round(auc(h["gap"], h["background"]), 4),
                "auc_gap_vs_roof": round(auc(h["gap"], h["roof"]), 4),
            }
        )
    record = {
        "dataset": args.data,
        "split": "val_tune",
        "created": timestamp(),
        "git": git_revision(),
        "gap_radius_px": radius,
        **counts,
        "results": results,
    }
    target = paths.out_root() / "metrics" / args.data / "phase2" / "gap_check_val_tune.json"
    write_json(target, record)
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
