"""A tiny fake of the official WHU 0.2 m release (docs/DATA.md §3.1): 1024 × 1024 LZW TIFF tiles with numeric names,
binary labels (0/255; an all-zero label is stored as int8, as in the release) and the official COCO files with one
polygon per building (category 1, image id = numeric stem, file names in upper case).

Tiles are synthetic scenes (tests/fixtures/synthetic.py) with one building group per 512² quadrant, so the touching
houses of the ``row`` group form one blob in the label but stay separate buildings in the COCO file. The listed
quadrants are emptied to exercise the quadrant filter; tile 10002 has no building at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from shapely.geometry import Polygon

from tests.fixtures.synthetic import Building, make_scene

TILE_SIZE = 1024
# folder -> tile stem -> quadrants emptied of buildings (0 top-left, 1 top-right, 2 bottom-left, 3 bottom-right)
TILES = {
    "train": {"10000": (), "10001": (3,), "10002": (0, 1, 2, 3)},
    "val": {"20000": (0, 1), "20001": ()},
    "test": {"3000": (2,), "3100": ()},  # part 310 is left out by the conversion
}
COCO_NAMES = {"train": "train", "val": "validation", "test": "test"}


@dataclass
class FakeTile:
    image: np.ndarray  # (1024, 1024, 3) uint8
    label: np.ndarray  # (1024, 1024) uint8, 0 or 255
    buildings: list[Building]  # buildings left after emptying quadrants


def quadrant_of(building: Building, size: int = TILE_SIZE) -> int:
    cx, cy = building.polygon.mean(axis=0)
    return 2 * int(cy >= size // 2) + int(cx >= size // 2)


def make_tile(seed: int, empty: tuple[int, ...] = (), size: int = TILE_SIZE) -> FakeTile:
    scene = make_scene(seed, size=size)
    kept = [b for b in scene.buildings if quadrant_of(b, size) not in empty]
    label = (np.isin(scene.instance_map, [b.id for b in kept]) * 255).astype(np.uint8)
    return FakeTile(scene.image, label, kept)


def make_fake_whu(data_root: Path) -> dict[str, dict[str, FakeTile]]:
    """Write data_root/whu/raw_0.2/<folder>/{image,label}/<stem>.tif and data_root/whu/official_coco_0.2/*.json;
    return the tiles by folder and stem."""
    tiles: dict[str, dict[str, FakeTile]] = {}
    seed = 100
    for folder, stems in TILES.items():
        tiles[folder] = {}
        images, annotations = [], []
        for stem, empty in stems.items():
            tile = make_tile(seed, empty)
            seed += 1
            tiles[folder][stem] = tile
            label = tile.label if tile.label.any() else tile.label.astype(np.int8)
            for kind, array, photometric in (("image", tile.image, "rgb"), ("label", label, "minisblack")):
                target = data_root / "whu" / "raw_0.2" / folder / kind
                target.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(target / f"{stem}.tif", array, photometric=photometric, compression="lzw")
            images.append({"id": int(stem), "file_name": f"{stem}.TIF", "width": TILE_SIZE, "height": TILE_SIZE})
            for b in tile.buildings:
                (x0, y0), (x1, y1) = b.polygon.min(axis=0), b.polygon.max(axis=0)
                annotations.append(
                    {
                        "segmentation": [b.polygon.ravel().tolist()],
                        "area": float(Polygon(b.polygon).area),
                        "iscrowd": 0,
                        "image_id": int(stem),
                        "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                        "category_id": 1,
                        "id": len(annotations) + 1,
                    }
                )
        coco = {
            "images": images,
            "annotations": annotations,
            "categories": [{"supercategory": "building", "id": 1, "name": "building"}],
        }
        target = data_root / "whu" / "official_coco_0.2" / f"{COCO_NAMES[folder]}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(coco), encoding="utf-8")
    return tiles
