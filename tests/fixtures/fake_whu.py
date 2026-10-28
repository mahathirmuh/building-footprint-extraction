"""A tiny fake WHU 0.2 m download: raw RGB tiles and binary (0/255) building masks in Pix2Poly's folder layout,
built from 512 × 512 synthetic scenes (9 patches of 224 × 224 per tile)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage import io

from tests.fixtures.synthetic import make_scene

TILES = {"train": ["tile_1.tif", "tile_2.tif"], "val": ["tile_3.tif", "tile_4.tif"], "test": ["tile_5_1.tif"]}


def make_tile(seed: int, size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    scene = make_scene(seed, size=size)
    return scene.image, ((scene.instance_map > 0) * 255).astype(np.uint8)


def make_fake_whu(data_root: Path) -> None:
    """Write data_root/whu/raw_0.2/<folder>/{image,label}/<tile>.tif."""
    seed = 100
    for folder, names in TILES.items():
        for name in names:
            image, label = make_tile(seed)
            seed += 1
            for kind, array in (("image", image), ("label", label)):
                target = data_root / "whu" / "raw_0.2" / folder / kind
                target.mkdir(parents=True, exist_ok=True)
                io.imsave(str(target / name), array, check_contrast=False)
