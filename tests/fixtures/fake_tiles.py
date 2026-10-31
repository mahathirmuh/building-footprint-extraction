"""Fake CrowdAI tiles laid out as the training dataset reads them: images, instance cache with index.csv, prior cache
and split lists, under DENSEPOLY_DATA. Tiles are crops of synthetic scenes, small enough for a network on CPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from densepoly import paths
from densepoly.priors.cache import write_prior
from densepoly.utils.io import write_lines, write_png
from tests.fixtures.synthetic import make_scene


def write_tiles(splits_dir: Path, tiles_per_split: dict[str, int], size: int = 120) -> dict[str, list[int]]:
    """One size × size tile per scene, consecutive seeds and image ids over the splits, the two parts of val_tune in one
    cache folder as in the real layout. The prior is a horizontal ramp (MBI) and a vertical one (gap). Returns the image
    ids per split."""
    ids, index, image_id = {}, {}, 0
    for split, count in tiles_per_split.items():
        names = []
        ids[split] = []
        for _ in range(count):
            image_id += 1
            scene = make_scene(image_id)
            name = f"{image_id:012d}.png"
            folder = paths.image_dir("crowdai", split)
            folder.mkdir(parents=True, exist_ok=True)
            Image.fromarray(scene.image[:size, :size]).save(folder / name)
            write_png(paths.instance_cache("crowdai", split, image_id), scene.instance_map[:size, :size])
            ramp = np.linspace(0, 255, size).astype(np.uint8)
            prior = np.dstack([np.tile(ramp, (size, 1)), np.tile(ramp[:, None], (1, size))])
            write_prior(paths.prior_cache("crowdai", split, image_id), prior)
            names.append(name)
            ids[split].append(image_id)
            index.setdefault(paths.cache_split(split), []).append(f"{name},{image_id},1")
        write_lines(paths.split_list("crowdai", split, splits_dir), names)
    for folder, rows in index.items():
        write_lines(paths.instance_dir("crowdai", folder) / "index.csv", ["file_name,image_id,instances", *rows])
    return ids
