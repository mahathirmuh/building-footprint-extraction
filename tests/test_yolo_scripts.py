"""Export of the FULL+Y detector's training set on fake tiles (scripts/prepare_yolo.py); no Ultralytics needed."""

import importlib.util

import numpy as np
import pytest

from densepoly import paths
from densepoly.postproc.yolo_seeds import obb_label_lines
from densepoly.utils.io import read_json, read_png
from tests.fixtures.fake_tiles import write_tiles


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, paths.REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tiles(tmp_path, monkeypatch):
    monkeypatch.setenv("DENSEPOLY_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("DENSEPOLY_OUT", str(tmp_path / "outputs"))
    folder = tmp_path / "splits"
    folder.mkdir()
    return folder, write_tiles(folder, {"train_dedup": 3, "val_select": 2})


def test_export_writes_one_label_file_and_one_image_link_per_tile(tiles, capsys):
    folder, ids = tiles
    load_script("prepare_yolo").main(["--data", "crowdai", "--workers", "1", "--splits-dir", str(folder)])
    root = paths.yolo_dir("crowdai") / "dataset"
    report = read_json(root / "export.json")
    assert report["parts"]["train"]["tiles"] == 3 and report["parts"]["val"]["tiles"] == 2
    for part, split in (("train", "train_dedup"), ("val", "val_select")):
        for image_id in ids[split]:
            label = root / "labels" / part / f"{image_id}.txt"
            lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line]
            expected = obb_label_lines(read_png(paths.instance_cache("crowdai", split, image_id)))
            assert lines == expected and len(lines) > 0
            values = np.array([line.split()[1:] for line in lines], float)
            assert values.shape[1] == 8 and values.min() >= 0 and values.max() <= 1
            link = root / "images" / part / f"{image_id}.png"
            assert link.is_symlink() and link.resolve() == (paths.image_dir("crowdai", split) / f"{image_id:012d}.png").resolve()
    boxes = sum(len(obb_label_lines(read_png(paths.instance_cache("crowdai", "train_dedup", i)))) for i in ids["train_dedup"])
    assert report["parts"]["train"]["boxes"] == boxes
    yaml = (root / "data.yaml").read_text(encoding="utf-8")
    assert f"path: {root}" in yaml and "train: images/train" in yaml and "0: building" in yaml


def test_export_can_be_limited_and_repeated(tiles):
    folder, _ = tiles
    for _ in range(2):  # the second run replaces the links instead of failing on them
        load_script("prepare_yolo").main(["--data", "crowdai", "--workers", "1", "--limit", "1", "--splits-dir", str(folder)])
    report = read_json(paths.yolo_dir("crowdai") / "dataset" / "export.json")
    assert report["limit"] == 1 and report["parts"]["train"]["tiles"] == 1 and report["parts"]["val"]["tiles"] == 1
