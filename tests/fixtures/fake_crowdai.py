"""A tiny fake CrowdAI download built from synthetic scenes, for end-to-end tests of the Phase 1 scripts.

train: scenes 0-9 (t00.png ... t09.png) plus an exact 90° rotation of t00 (t00_rot.png) and an exact horizontal flip
of t01 (t01_flip.png): 12 images in 10 duplicate clusters.
val: scene 20 (v20.png, image id 1), scene 21 (v21.png, id 2), a transposed copy of train t03 (v_leak.png, leaked
from train) and a vertically flipped copy of v20 (v20_copy.png): the clean test is v20.png and v21.png.
Copies carry no annotations. Images are PNG so that the copies are pixel-exact; real CrowdAI tiles are JPEG.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tests.fixtures.synthetic import Scene, make_scene, to_coco

T = Image.Transpose


def make_fake_crowdai(data_root: Path) -> dict[str, dict[str, Scene]]:
    """Write data_root/crowdai/{train,val}/{images/, annotation.json}; return the annotated scenes by file name."""
    scenes = {
        "train": {f"t{s:02d}.png": make_scene(s) for s in range(10)},
        "val": {"v20.png": make_scene(20), "v21.png": make_scene(21)},
    }
    copies = {
        "train": {"t00_rot.png": ("t00.png", T.ROTATE_90), "t01_flip.png": ("t01.png", T.FLIP_LEFT_RIGHT)},
        "val": {"v_leak.png": ("t03.png", T.TRANSPOSE), "v20_copy.png": ("v20.png", T.FLIP_TOP_BOTTOM)},
    }
    every_scene = {**scenes["train"], **scenes["val"]}
    for source in ("train", "val"):
        folder = data_root / "crowdai" / source / "images"
        folder.mkdir(parents=True)
        for name, scene in scenes[source].items():
            Image.fromarray(scene.image).save(folder / name)
        for name, (original, op) in copies[source].items():
            Image.fromarray(every_scene[original].image).transpose(op).save(folder / name)
        coco = to_coco(list(scenes[source].values()), file_names=list(scenes[source]))
        for name in copies[source]:
            coco["images"].append({"id": len(coco["images"]) + 1, "file_name": name, "height": 300, "width": 300})
        (data_root / "crowdai" / source / "annotation.json").write_text(json.dumps(coco), encoding="utf-8")
    return scenes
