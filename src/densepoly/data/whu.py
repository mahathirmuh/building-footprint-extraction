"""WHU aerial buildings, protocol A (docs/DATA.md §3.2): Pix2Poly's conversion of the 0.2 m rasters into 224 × 224
COCO patches, ported without changing its logic.

Adapted from Pix2Poly ``data_preprocess/whu_buildings_to_coco.py`` (commit 4e4c003,
https://github.com/yeshwanth95/Pix2Poly), which borrows HiSup ``tools/inria_to_coco.py``.
Copyright (c) 2025 Yeshwanth Kumar Adimoolam; Copyright (c) 2019 Bowen Xu, Jiakun Xu, Nan Xue and Gui-Song Xia.
MIT License, see third_party/licenses/. The original file is kept verbatim in third_party/pix2poly/, and
tests/test_whu.py checks that this port returns identical polygons and patch grids.

Wrapper changes only:
- tiles are read in sorted order (Pix2Poly uses os.listdir, whose order depends on the file system), so ids and
  patch names are reproducible; only the numbering can differ from a Pix2Poly run, never the polygons;
- files other than .tif in the label folder are ignored, and paths come from DENSEPOLY_DATA;
- left out: the rotation augmentation (its rotation_list is empty, so it never runs), the lt_crop call (a no-op on
  224 × 224 patches) and the per-tile GeoJSON export (the COCO file does not use it).

Behaviour kept on purpose:
- connected components use 8-connectivity, so touching (even diagonally touching) buildings become one instance;
- contours run through boundary pixel centres (cv2.findContours, CHAIN_APPROX_SIMPLE) clipped to [0, size - 1],
  so a polygon is about half a pixel smaller than its mask on every side;
- holes larger than 10 px² become extra rings of the same annotation (pycocotools fills them again);
- a patch with at most 5 building pixels gets one empty "negative" annotation, while a patch with more pixels whose
  polygons are all filtered out (bbox side <= 5 px) gets no annotation at all;
- image and annotation ids start at 0 in every split.
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Sequence
from itertools import groupby
from pathlib import Path

import cv2
import numpy as np
from shapely import STRtree
from shapely.geometry import Polygon, mapping
from skimage import io
from skimage.measure import label as ski_label
from skimage.measure import regionprops

from densepoly.data.crowdai import valid_rings
from densepoly.utils.io import write_json

PATCH_SIZE = 224
PATCH_OVERLAP = 34  # ~15% of the patch size
SIMPLIFY_TOLERANCE = 1.8
CATEGORY_ID = 100
INFO = {"district": "Christchurch, NZ", "description": "WHU building footprints", "contributor": "whu"}


def clip_by_bound(poly: np.ndarray, im_h: int, im_w: int) -> np.ndarray:
    """Clip (x, y) coordinates to x in [0, im_w - 1] and y in [0, im_h - 1]."""
    p_x = np.clip(poly[:, 0], 0.0, im_w - 1)
    p_y = np.clip(poly[:, 1], 0.0, im_h - 1)
    return np.concatenate((p_x[:, np.newaxis], p_y[:, np.newaxis]), axis=1)


def crop2patch(im_p: tuple[int, ...], p_h: int, p_w: int, p_overlap: int) -> list[list[int]]:
    """Patches [x_ul, y_ul, width, height] with stride p - overlap; the last row/column is flush with the border."""
    im_h, im_w, _ = im_p
    x = np.append(np.arange(0, im_w - p_w, p_w - p_overlap), im_w - p_w)
    y = np.append(np.arange(0, im_h - p_h, p_h - p_overlap), im_h - p_h)
    xs, ys = np.meshgrid(x, y)
    return [[i, j, p_w, p_h] for i, j in zip(xs.flatten(), ys.flatten(), strict=True)]


def bmask_to_poly(b_im: np.ndarray, simplify_ind: int, tolerance: float = SIMPLIFY_TOLERANCE) -> list[Polygon]:
    """Binary mask -> polygons: one per 8-connected component (holes > 10 px² kept), contours traced by cv2 on the
    component padded by 1 px, then shapely simplify(tolerance, preserve_topology=False), which may split a polygon."""
    polygons = []
    for prop in regionprops(ski_label(b_im > 0)):
        prop_mask = np.zeros_like(b_im)
        prop_mask[prop.coords[:, 0], prop.coords[:, 1]] = 1
        padded_binary_mask = np.pad(prop_mask, pad_width=1, mode="constant", constant_values=0)
        contours, hierarchy = cv2.findContours(padded_binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 1:
            intp = []
            for contour, h in zip(contours, hierarchy[0], strict=True):
                contour = np.array([c.reshape(-1).tolist() for c in contour])
                contour -= 1  # undo the padding
                contour = clip_by_bound(contour, b_im.shape[0], b_im.shape[1])
                if len(contour) > 3:
                    closed_c = np.concatenate((contour, contour[0].reshape(-1, 2)))
                    if h[3] < 0:
                        extp = [tuple(i) for i in closed_c]
                    elif cv2.contourArea(closed_c.astype(int)) > 10:
                        intp.append([tuple(i) for i in closed_c])
            poly = Polygon(extp, intp)
            if simplify_ind:
                poly = poly.simplify(tolerance=tolerance, preserve_topology=False)
                polygons.extend([poly] if isinstance(poly, Polygon) else poly.geoms)
        elif len(contours) == 1:
            contour = np.array([c.reshape(-1).tolist() for c in contours[0]])
            contour -= 1
            contour = clip_by_bound(contour, b_im.shape[0], b_im.shape[1])
            if len(contour) > 3:
                closed_c = np.concatenate((contour, contour[0].reshape(-1, 2)))
                poly = Polygon(closed_c)
                if simplify_ind:
                    poly = poly.simplify(tolerance=tolerance, preserve_topology=False)
                polygons.extend([poly] if isinstance(poly, Polygon) else poly.geoms)
    return polygons


def patch_annotations(p_gt: np.ndarray, image_id: int, first_ann_id: int) -> list[dict]:
    """COCO annotations of one patch exactly as Pix2Poly writes them, with ids from first_ann_id."""
    anns = []
    if np.sum(p_gt > 0) > 5:
        for poly in bmask_to_poly(p_gt, 1):
            p_area = round(poly.area, 2)
            if p_area > 0:
                minx, miny, maxx, maxy = poly.bounds
                p_bbox = [minx, miny, maxx - minx, maxy - miny]
                if p_bbox[2] > 5 and p_bbox[3] > 5:
                    p_seg = [np.asarray(part).ravel().tolist() for part in mapping(poly)["coordinates"]]
                    anns.append(
                        {
                            "id": first_ann_id + len(anns),
                            "image_id": image_id,
                            "segmentation": p_seg,
                            "area": p_area,
                            "bbox": p_bbox,
                            "category_id": CATEGORY_ID,
                            "iscrowd": 0,
                        }
                    )
    else:  # negative sample
        anns.append(
            {
                "id": first_ann_id,
                "image_id": image_id,
                "segmentation": [],
                "area": 0.0,
                "bbox": [],
                "category_id": CATEGORY_ID,
                "iscrowd": 0,
            }
        )
    return anns


def tile_name(label_file: str, split: str) -> str:
    """Tile stem as Pix2Poly parses it: the first two digit/non-digit runs of the file name, four for test."""
    runs = ["".join(g) for _, g in groupby(label_file, key=lambda ch: ch.isdigit())]
    return "".join(runs[:4] if split == "test" else runs[:2])


def convert_split(raw_dir: Path, out_dir: Path, split: str, save_images: bool = True) -> tuple[dict, list[str]]:
    """Convert raw_dir/{image,label}/*.tif into out_dir/{images/*.tif, annotation.json}; return counts and patch
    file names in id order."""
    output = {
        "info": dict(INFO),
        "categories": [{"id": CATEGORY_ID, "name": "building"}],
        "images": [],
        "annotations": [],
    }
    images_out = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)
    tiles = sorted(f for f in os.listdir(raw_dir / "label") if f.lower().endswith(".tif"))
    counts = {"tiles": len(tiles), "negative_patches": 0, "patches_without_annotation": 0, "multi_ring_annotations": 0}
    for label_file in tiles:
        name = tile_name(label_file, split)
        image_data = io.imread(str(raw_dir / "image" / f"{name}.tif"))
        gt_im_data = io.imread(str(raw_dir / "label" / f"{name}.tif"))
        for x_ul, y_ul, _, _ in crop2patch(image_data.shape, PATCH_SIZE, PATCH_SIZE, PATCH_OVERLAP):
            p_gt = gt_im_data[y_ul : y_ul + PATCH_SIZE, x_ul : x_ul + PATCH_SIZE]
            p_im = image_data[y_ul : y_ul + PATCH_SIZE, x_ul : x_ul + PATCH_SIZE, :]
            image_id = len(output["images"])
            anns = patch_annotations(p_gt, image_id, len(output["annotations"]))
            counts["negative_patches"] += len(anns) == 1 and not anns[0]["segmentation"]
            counts["patches_without_annotation"] += not anns
            counts["multi_ring_annotations"] += sum(len(a["segmentation"]) > 1 for a in anns)
            output["annotations"].extend(anns)
            p_name = f"{name}-{image_id}.tif"
            output["images"].append({"id": image_id, "file_name": p_name, "width": PATCH_SIZE, "height": PATCH_SIZE})
            if save_images:
                io.imsave(str(images_out / p_name), p_im, check_contrast=False)
    write_json(out_dir / "annotation.json", output, indent=None)
    counts["patches"] = len(output["images"])
    counts["building_annotations"] = sum(1 for a in output["annotations"] if a["segmentation"])
    return counts, [img["file_name"] for img in output["images"]]


def split_val_by_tile(patch_names: Sequence[str], tune_sub_size: int = 1000, seed: int = 0) -> tuple[list, list]:
    """val_tune_sub and val_select for WHU. Patches of one tile overlap by 34 px, so whole tiles are drawn in random
    order (seed) into val_tune_sub until it holds at least tune_sub_size patches; the other tiles form val_select."""
    tiles: dict[str, list[str]] = defaultdict(list)
    for name in patch_names:
        tiles[name.rsplit("-", 1)[0]].append(name)
    order = sorted(tiles)
    sub: list[str] = []
    for i in np.random.default_rng(seed).permutation(len(order)):
        if len(sub) >= tune_sub_size:
            break
        sub.extend(tiles[order[i]])
    chosen = set(sub)
    return [n for n in patch_names if n in chosen], [n for n in patch_names if n not in chosen]


def touching_pairs(anns: Sequence[dict], max_distance: float = 1.0) -> int:
    """Pairs of GT polygons in the same image that overlap, touch or are closer than max_distance px (exterior rings);
    used to check whether touching buildings are separate instances (docs/DATA.md §3.4)."""
    by_image: dict[int, list[Polygon]] = defaultdict(list)
    for ann in anns:
        rings = valid_rings(ann["segmentation"]) if isinstance(ann["segmentation"], list) else []
        if rings:
            by_image[ann["image_id"]].append(Polygon(np.asarray(rings[0], float).reshape(-1, 2)).buffer(0))
    pairs = 0
    for polys in by_image.values():
        tree = STRtree(polys)
        for i, poly in enumerate(polys):
            for j in tree.query(poly, predicate="dwithin", distance=max_distance):
                pairs += j > i and polys[j].distance(poly) < max_distance
    return int(pairs)
