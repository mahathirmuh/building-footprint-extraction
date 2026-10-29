"""Instance maps L (docs/METHOD.md §3.1): the COCO annotations of the main ground truth rasterised by decreasing area
(densepoly.data.crowdai.rasterize_instances), cached as one uint16 PNG per image so that the training loader never
reads the COCO files. WHU patches use the main ground truth (official buildings), never Pix2Poly's p2p one."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pycocotools.coco import COCO

from densepoly import paths
from densepoly.data.crowdai import load_coco, rasterize_instances
from densepoly.utils.io import read_lines, read_png, write_lines, write_png


def cache_instances(dataset: str, split: str, splits_dir: Path | None = None, coco: COCO | None = None) -> dict:
    """Write the instance map of every image of a split and an index.csv (file_name, image_id, instances); return
    the rasterisation totals. coco: the split's annotation file when already loaded (it can be large)."""
    coco = coco or load_coco(paths.annotation_file(dataset, split))
    by_name = {img["file_name"]: img for img in coco.dataset["images"]}
    totals = {"images": 0, "instances": 0, "skipped": 0, "empty": 0, "overlap_px": 0, "hidden": 0}
    index = ["file_name,image_id,instances"]
    for name in read_lines(paths.split_list(dataset, split, splits_dir)):
        img = by_name[name]
        result = rasterize_instances(coco.imgToAnns.get(img["id"], []), img["height"], img["width"])
        write_png(paths.instance_cache(dataset, split, img["id"]), result.instance_map)
        index.append(f"{name},{img['id']},{len(result.annotation_ids)}")
        totals["images"] += 1
        totals["instances"] += len(result.annotation_ids)
        totals["skipped"] += result.n_skipped  # includes WHU's negative samples (empty segmentation)
        totals["empty"] += result.n_empty
        totals["overlap_px"] += result.overlap_px
        totals["hidden"] += result.n_hidden
    write_lines(paths.instance_dir(dataset, split) / "index.csv", index)
    return totals


def load_instances(dataset: str, split: str, image_id: int | str) -> np.ndarray:
    """Cached (H, W) uint16 instance map of one image."""
    return read_png(paths.instance_cache(dataset, split, image_id))
