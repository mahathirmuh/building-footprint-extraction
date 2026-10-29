"""WHU aerial buildings (docs/DATA.md §3.2): Pix2Poly's conversion of the 0.2 m rasters into 224 × 224 COCO patches,
ported without changing its logic, plus a main ground truth that keeps the official buildings apart.

Adapted from Pix2Poly ``data_preprocess/whu_buildings_to_coco.py`` (commit 4e4c003,
https://github.com/yeshwanth95/Pix2Poly), which borrows HiSup ``tools/inria_to_coco.py``.
Copyright (c) 2025 Yeshwanth Kumar Adimoolam; Copyright (c) 2019 Bowen Xu, Jiakun Xu, Nan Xue and Gui-Song Xia.
MIT License, see third_party/licenses/. The original file is kept verbatim in third_party/pix2poly/, and
tests/test_whu.py checks that this port returns identical polygons and patch grids.

Tiles. The Pix2Poly paper uses 512 × 512 tiles at 0.2 m (9,420 train, 1,537 val, 3,848 test); the official release
has 1024 × 1024 tiles with numeric names. Each official tile is cut into four 512² quadrants and the quadrants with
building pixels are kept: 9,418 train and 1,529 val tiles. The test keeps only part 300 of the test area (3,943
tiles), the closest to Pix2Poly's test, which cannot be rebuilt exactly (docs/progress.md, 29 Oct 2026). Every kept
quadrant then goes through Pix2Poly's loop like one of its tiles.

Two ground truths share the same patches (option R, 29 Oct 2026):
- main: the official COCO buildings, rasterised per tile, each building's mask going through Pix2Poly's polygon
  steps (bmask_to_poly, the bbox filter, the negative-sample rule), so touching buildings stay separate;
- ``p2p``: Pix2Poly's ground truth from the label raster, only for comparison with the Pix2Poly paper.

Wrapper changes only:
- tiles are read in sorted order (Pix2Poly uses os.listdir, whose order depends on the file system) and named after
  the file stem; Pix2Poly parses names from digit runs, which fails on the official numeric names, and for every name
  its parser accepts the stem is the same name. Only the numbering can differ from a Pix2Poly run, never the polygons;
- tiles are converted in parallel and ids are assigned afterwards in tile order, as in a serial run;
- patch images are not written: they are exact crops of the official tiles at the offsets kept in the patch index;
- labels with negative values are rejected (the release stores its all-zero tiles as int8);
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

import multiprocessing
import os
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tifffile
from shapely import STRtree
from shapely.geometry import Polygon, mapping
from skimage import io
from skimage.measure import label as ski_label
from skimage.measure import regionprops

from densepoly.data.crowdai import rasterize_instances, valid_rings

PATCH_SIZE = 224
PATCH_OVERLAP = 34  # ~15% of the patch size
SIMPLIFY_TOLERANCE = 1.8
CATEGORY_ID = 100
QUADRANT = 512  # side of Pix2Poly's WHU tiles
TEST_PART = "300"  # official test tiles 300* (1,540 tiles); part 310* (680 tiles) is left out
OFFICIAL_COCO_NAMES = {"train": "train", "val": "validation", "test": "test"}  # official COCO file per folder
PATCH_INDEX_COLUMNS = ("file_name", "image_id", "tile", "quadrant", "x", "y")  # x, y: offset in the official tile
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


def _polygon_annotations(mask: np.ndarray, image_id: int, first_ann_id: int) -> list[dict]:
    """Pix2Poly's annotations of the polygons of one binary mask (area > 0, both bbox sides > 5 px)."""
    anns = []
    for poly in bmask_to_poly(mask, 1):
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
    return anns


def _negative_sample(image_id: int, ann_id: int) -> dict:
    return {
        "id": ann_id,
        "image_id": image_id,
        "segmentation": [],
        "area": 0.0,
        "bbox": [],
        "category_id": CATEGORY_ID,
        "iscrowd": 0,
    }


def patch_annotations(p_gt: np.ndarray, image_id: int, first_ann_id: int) -> list[dict]:
    """COCO annotations of one patch exactly as Pix2Poly writes them, with ids from first_ann_id."""
    if np.sum(p_gt > 0) > 5:
        return _polygon_annotations(p_gt, image_id, first_ann_id)
    return [_negative_sample(image_id, first_ann_id)]


def instance_patch_annotations(p_inst: np.ndarray, image_id: int, first_ann_id: int) -> list[dict]:
    """Main ground truth of one patch: Pix2Poly's polygon steps applied to every building of the instance map on its
    own, so touching buildings stay apart. The negative-sample rule counts all building pixels, as in Pix2Poly."""
    if np.sum(p_inst > 0) <= 5:
        return [_negative_sample(image_id, first_ann_id)]
    anns = []
    for value in np.unique(p_inst[p_inst > 0]):
        mask = (p_inst == value).astype(np.uint8)
        anns.extend(_polygon_annotations(mask, image_id, first_ann_id + len(anns)))
    return anns


def quadrants(label: np.ndarray, size: int = QUADRANT) -> list[tuple[int, int, int]]:
    """(index, y0, x0) of the size × size quadrants that contain building pixels, row by row."""
    height, width = label.shape[:2]
    if height % size or width % size:
        raise ValueError(f"a {height} × {width} tile does not split into {size}² quadrants")
    corners = [(y, x) for y in range(0, height, size) for x in range(0, width, size)]
    return [(q, y, x) for q, (y, x) in enumerate(corners) if (label[y : y + size, x : x + size] > 0).any()]


def official_by_tile(coco: dict) -> dict[str, list[dict]]:
    """Official COCO annotations grouped by tile stem (file name without extension), every tile listed."""
    stems = {img["id"]: os.path.splitext(img["file_name"])[0] for img in coco["images"]}
    tiles: dict[str, list[dict]] = {stem: [] for stem in stems.values()}
    for ann in coco["annotations"]:
        tiles[stems[ann["image_id"]]].append(ann)
    return tiles


def convert_tile(raw_dir: Path, file_name: str, official: list[dict] | None) -> dict:
    """Patches of one official tile: its quadrants with buildings, Pix2Poly's patch grid inside each quadrant, and the
    annotations of both ground truths (local ids: image 0, annotations from 0; convert_split renumbers them).
    official: the tile's official COCO annotations, or None to skip the main ground truth."""
    label = io.imread(str(raw_dir / "label" / file_name))
    if label.min() < 0:
        raise ValueError(f"{file_name}: negative label values")
    label = label.astype(np.uint8)
    with tifffile.TiffFile(raw_dir / "image" / file_name) as tif:
        image_shape = tuple(tif.pages[0].shape)
    if image_shape[:2] != label.shape:
        raise ValueError(f"{file_name}: image {image_shape} and label {label.shape} differ")
    raster = None if official is None else rasterize_instances(official, *label.shape)
    patches = []
    for q, y0, x0 in quadrants(label):
        tile = label[y0 : y0 + QUADRANT, x0 : x0 + QUADRANT]  # a Pix2Poly tile
        for x_ul, y_ul, _, _ in crop2patch((QUADRANT, QUADRANT, 3), PATCH_SIZE, PATCH_SIZE, PATCH_OVERLAP):
            p_gt = tile[y_ul : y_ul + PATCH_SIZE, x_ul : x_ul + PATCH_SIZE]
            x, y = x0 + int(x_ul), y0 + int(y_ul)
            main = None
            if raster is not None:
                main = instance_patch_annotations(raster.instance_map[y : y + PATCH_SIZE, x : x + PATCH_SIZE], 0, 0)
            patches.append({"quadrant": q, "x": x, "y": y, "p2p": patch_annotations(p_gt, 0, 0), "main": main})
    official_stats = None
    if raster is not None:
        official_stats = {
            "annotations": len(official),
            "skipped": raster.n_skipped + raster.n_empty,
            "overlap_px": raster.overlap_px,
            "hidden": raster.n_hidden,
        }
    return {"patches": patches, "official": official_stats}


@dataclass
class Conversion:
    images: list[dict]  # shared by both ground truths
    main: list[dict] | None  # annotations of the main ground truth (official buildings)
    p2p: list[dict]  # annotations of Pix2Poly's ground truth
    patches: list[dict]  # PATCH_INDEX_COLUMNS of every patch
    counts: dict


def coco_dict(images: list[dict], annotations: list[dict]) -> dict:
    """COCO file in Pix2Poly's layout."""
    categories = [{"id": CATEGORY_ID, "name": "building"}]
    return {"info": dict(INFO), "categories": categories, "images": images, "annotations": annotations}


def _annotation_counts(anns: list[dict], n_images: int) -> dict:
    return {
        "building_annotations": sum(1 for a in anns if a["segmentation"]),
        "negative_patches": sum(1 for a in anns if not a["segmentation"]),
        "patches_without_annotation": n_images - len({a["image_id"] for a in anns}),
        "multi_ring_annotations": sum(len(a["segmentation"]) > 1 for a in anns),
    }


def convert_split(
    raw_dir: Path, split: str, official: dict[str, list[dict]] | None = None, workers: int = 1
) -> Conversion:
    """Convert raw_dir/{image,label}/<stem>.tif. official maps every tile stem to its official COCO annotations
    (official_by_tile); None skips the main ground truth. The test keeps the tiles of part TEST_PART only. Patch names
    are <stem>_<quadrant>-<image id>.tif."""
    files = sorted(f for f in os.listdir(raw_dir / "label") if f.lower().endswith(".tif"))
    used = [f for f in files if split != "test" or f.startswith(TEST_PART)]
    if official is not None and (missing := [f for f in used if os.path.splitext(f)[0] not in official]):
        raise KeyError(f"tiles without an entry in the official COCO file: {missing[:5]}")
    jobs = [(raw_dir, f, None if official is None else official[os.path.splitext(f)[0]]) for f in used]
    if workers > 1:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            results = pool.starmap(convert_tile, jobs, chunksize=4)
    else:
        results = [convert_tile(*job) for job in jobs]

    images, p2p, patches = [], [], []
    main = None if official is None else []
    official_totals: dict[str, int] = defaultdict(int)
    quadrants_kept = 0
    for (_, file_name, _), result in zip(jobs, results, strict=True):
        stem = os.path.splitext(file_name)[0]
        quadrants_kept += len({p["quadrant"] for p in result["patches"]})
        for patch in result["patches"]:
            image_id = len(images)
            name = f"{stem}_{patch['quadrant']}-{image_id}.tif"
            images.append({"id": image_id, "file_name": name, "width": PATCH_SIZE, "height": PATCH_SIZE})
            row = {"file_name": name, "image_id": image_id, "tile": stem}
            patches.append(row | {k: patch[k] for k in ("quadrant", "x", "y")})
            for target, anns in ((p2p, patch["p2p"]), (main, patch["main"])):
                if target is None:
                    continue
                for ann in anns:
                    ann["id"], ann["image_id"] = len(target), image_id
                    target.append(ann)
        for key, value in (result["official"] or {}).items():
            official_totals[key] += value
    counts = {
        "tiles": len(files),
        "tiles_used": len(used),
        "quadrants_kept": quadrants_kept,
        "patches": len(images),
        "p2p": _annotation_counts(p2p, len(images)),
    }
    if main is not None:
        counts["main"] = _annotation_counts(main, len(images))
        counts["official_in_used_tiles"] = dict(official_totals)
    return Conversion(images, main, p2p, patches, counts)


def split_val_by_tile(patch_names: Sequence[str], tune_sub_size: int = 1000, seed: int = 0) -> tuple[list, list]:
    """val_tune_sub and val_select for WHU. Patches of one quadrant overlap and the quadrants of an official tile touch,
    so whole official tiles (the stem in <stem>_<quadrant>-<id>.tif) are drawn in random order (seed) into
    val_tune_sub until it holds at least tune_sub_size patches; the other tiles form val_select."""
    tiles: dict[str, list[str]] = defaultdict(list)
    for name in patch_names:
        tiles[name.rsplit("-", 1)[0].rsplit("_", 1)[0]].append(name)
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
