"""End-to-end run of the Phase 1 scripts on a tiny fake CrowdAI download (tests/fixtures/fake_crowdai.py)."""

import importlib.util

import numpy as np
import pytest

from densepoly import paths
from densepoly.utils.io import read_json, read_lines, read_png
from tests.fixtures.fake_crowdai import make_fake_crowdai
from tests.fixtures.fake_whu import make_fake_whu


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, paths.REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase1_scripts_on_a_fake_download(tmp_path, monkeypatch):
    scenes = make_fake_crowdai(tmp_path / "data")
    monkeypatch.setenv("DENSEPOLY_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("DENSEPOLY_OUT", str(tmp_path / "outputs"))
    splits_dir = tmp_path / "splits"
    common = ["--splits-dir", str(splits_dir)]

    # deduplication: copies collapse, the leaked val image and the val-internal copy leave the test set
    load_script("dedup_crowdai").main(["--data", "crowdai", "--workers", "1", *common])
    names = ("train_dedup", "val_tune", "val_tune_sub", "val_select", "test")
    split = {name: read_lines(paths.split_list("crowdai", name, splits_dir)) for name in names}
    assert split["test"] == ["v20.png", "v21.png"]
    assert len(split["train_dedup"]) == 9 and len(split["val_tune"]) == 1
    assert not set(split["train_dedup"]) & set(split["val_tune"])
    assert {"t00_rot.png", "t01_flip.png"}.isdisjoint(split["train_dedup"] + split["val_tune"])
    report = read_json(splits_dir / "dedup_report.json")
    assert report["train"]["clusters"] == 10 and report["val"]["clean_test_clusters"] == 2
    assert (tmp_path / "outputs" / "figures" / "qa" / "dedup_clusters.png").is_file()
    assert len(read_json(tmp_path / "data" / "crowdai" / "hashes" / "train_aug.json")) == 12

    # download check and instance cache
    prepare = load_script("prepare_crowdai")
    prepare.main(["--stats", *common])
    stats = read_json(splits_dir / "crowdai_stats.json")
    assert (stats["train"]["images"], stats["train"]["images_without_annotations"]) == (12, 2)
    assert stats["val"]["missing_files"] == stats["val"]["extra_files"] == 0
    prepare.main(["--cache", "train_dedup", "val_tune", "test", *common])
    cached = read_png(paths.instance_cache("crowdai", "test", 1))  # v20.png has image id 1
    np.testing.assert_array_equal(cached, scenes["val"]["v20.png"].instance_map)
    assert read_json(splits_dir / "crowdai_raster_stats.json")["test"]["images"] == 2

    # density strata: test first, frozen afterwards, val_tune reuses its thresholds
    make_density = load_script("make_density")
    make_density.main(["--data", "crowdai", "--split", "test", *common])
    with pytest.raises(SystemExit):
        make_density.main(["--data", "crowdai", "--split", "test", *common])
    make_density.main(["--data", "crowdai", "--split", "val_tune", *common])
    test_strata = read_json(paths.density_file("crowdai", "test", splits_dir))
    tune_strata = read_json(paths.density_file("crowdai", "val_tune", splits_dir))
    assert tune_strata["thresholds"] == test_strata["thresholds"]
    assert sorted(i for ids in test_strata["main"]["strata"].values() for i in ids) == [1, 2]


def test_whu_scripts_on_a_fake_download(tmp_path, monkeypatch):
    make_fake_whu(tmp_path / "data")
    monkeypatch.setenv("DENSEPOLY_DATA", str(tmp_path / "data"))
    splits_dir = tmp_path / "splits"
    common = ["--splits-dir", str(splits_dir)]

    load_script("prepare_whu_pix2poly").main(common)
    split = {s: read_lines(paths.split_list("whu", s, splits_dir)) for s in ("train_dedup", "val_tune", "test")}
    assert [len(split[s]) for s in ("train_dedup", "val_tune", "test")] == [18, 18, 9]  # 9 patches per 512² tile
    assert read_lines(paths.split_list("whu", "val_tune_sub", splits_dir)) == split["val_tune"]  # fewer than 1,000
    stats = read_json(splits_dir / "whu_stats.json")
    assert stats["test"]["patches"] == 9 and stats["test"]["tiles"] == 1
    test_coco = read_json(paths.annotation_file("whu", "test"))
    assert [img["file_name"] for img in test_coco["images"]] == split["test"]

    load_script("prepare_whu_pix2poly").main(["--official-coco", str(paths.annotation_file("whu", "test")), *common])
    assert read_json(splits_dir / "whu_stats.json")["official_coco"]["touching_pairs"] == 0  # CC never touch

    make_density = load_script("make_density")
    make_density.main(["--data", "whu", "--split", "test", *common])
    make_density.main(["--data", "whu", "--split", "val_tune", *common])
    strata = read_json(paths.density_file("whu", "test", splits_dir))
    assert strata["contact_radius_px"] == 3 and sum(strata["main"]["sizes"].values()) == 9
