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

    # deduplication: copies collapse into 10 train clusters, one of which is held out as the test; the leaked val
    # image and the val-internal copy leave val_clean
    load_script("dedup_crowdai").main(["--data", "crowdai", "--workers", "1", *common])
    names = ("train_dedup", "val_tune", "val_tune_sub", "val_select", "test", "val_clean")
    split = {name: read_lines(paths.split_list("crowdai", name, splits_dir)) for name in names}
    assert split["val_clean"] == ["v20.png", "v21.png"]
    assert [len(split[s]) for s in ("train_dedup", "val_tune", "test")] == [8, 1, 1]
    representatives = split["train_dedup"] + split["val_tune"] + split["test"]
    assert len(set(representatives)) == 10 and {"t00_rot.png", "t01_flip.png"}.isdisjoint(representatives)
    report = read_json(splits_dir / "dedup_report.json")
    assert report["train"]["clusters"] == 10 and report["val"]["clean_clusters"] == 2
    clean_clusters = read_json(splits_dir / "crowdai_val_clean_clusters.json")
    assert clean_clusters == {"v20.png": ["v20.png", "v20_copy.png"], "v21.png": ["v21.png"]}
    assert (tmp_path / "outputs" / "figures" / "qa" / "dedup_clusters.png").is_file()
    assert len(read_json(tmp_path / "data" / "crowdai" / "hashes" / "train_aug.json")) == 12

    # download check and instance cache; the held-out test is a train tile (train image ids follow t00 ... t09)
    prepare = load_script("prepare_crowdai")
    prepare.main(["--stats", *common])
    stats = read_json(splits_dir / "crowdai_stats.json")
    assert (stats["train"]["images"], stats["train"]["images_without_annotations"]) == (12, 2)
    assert stats["val"]["missing_files"] == stats["val"]["extra_files"] == 0
    prepare.main(["--cache", "train_dedup", "val_tune", "test", "val_clean", *common])
    cached = read_png(paths.instance_cache("crowdai", "val_clean", 1))  # v20.png has image id 1
    np.testing.assert_array_equal(cached, scenes["val"]["v20.png"].instance_map)
    held_out = split["test"][0]
    held_out_id = list(scenes["train"]).index(held_out) + 1
    cached = read_png(paths.instance_cache("crowdai", "test", held_out_id))
    np.testing.assert_array_equal(cached, scenes["train"][held_out].instance_map)
    raster = read_json(splits_dir / "crowdai_raster_stats.json")
    assert (raster["test"]["images"], raster["val_clean"]["images"]) == (1, 2)

    # density strata: test first, frozen afterwards; val_tune and val_clean reuse its thresholds
    make_density = load_script("make_density")
    make_density.main(["--data", "crowdai", "--split", "test", *common])
    with pytest.raises(SystemExit):
        make_density.main(["--data", "crowdai", "--split", "test", *common])
    test_strata = read_json(paths.density_file("crowdai", "test", splits_dir))
    assert sorted(i for ids in test_strata["main"]["strata"].values() for i in ids) == [held_out_id]
    for name in ("val_tune", "val_clean"):
        make_density.main(["--data", "crowdai", "--split", name, *common])
        assert read_json(paths.density_file("crowdai", name, splits_dir))["thresholds"] == test_strata["thresholds"]
    clean_strata = read_json(paths.density_file("crowdai", "val_clean", splits_dir))
    assert sorted(i for ids in clean_strata["main"]["strata"].values() for i in ids) == [1, 2]


def test_whu_scripts_on_a_fake_download(tmp_path, monkeypatch):
    tiles = make_fake_whu(tmp_path / "data")
    monkeypatch.setenv("DENSEPOLY_DATA", str(tmp_path / "data"))
    splits_dir = tmp_path / "splits"
    common = ["--splits-dir", str(splits_dir)]

    # 9 patches per kept 512² quadrant: train 4 + 3 quadrants, val 2 + 4, test 3 (part 300 only)
    load_script("prepare_whu_pix2poly").main(["--workers", "1", *common])
    split = {s: read_lines(paths.split_list("whu", s, splits_dir)) for s in ("train_dedup", "val_tune", "test")}
    assert [len(split[s]) for s in ("train_dedup", "val_tune", "test")] == [63, 54, 27]
    assert read_lines(paths.split_list("whu", "val_tune_sub", splits_dir)) == split["val_tune"]  # fewer than 1,000
    stats = read_json(splits_dir / "whu_stats.json")
    assert (stats["test"]["tiles_used"], stats["train"]["quadrants_kept"]) == (1, 7)
    main, p2p = (read_json(paths.annotation_file("whu", "test", variant)) for variant in (None, "p2p"))
    assert main["images"] == p2p["images"] and [img["file_name"] for img in main["images"]] == split["test"]
    index = read_lines(paths.patch_index("whu", "test"))
    assert index[0] == "file_name,image_id,tile,quadrant,x,y" and len(index) == 1 + 27

    # the official COCO keeps the touching houses of every row group apart (both test tiles are in the file)
    official_test = tmp_path / "data" / "whu" / "official_coco_0.2" / "test.json"
    load_script("prepare_whu_pix2poly").main(["--official-coco", str(official_test), *common])
    shared_walls = sum(len(b.neighbors) for tile in tiles["test"].values() for b in tile.buildings) // 2
    assert read_json(splits_dir / "whu_stats.json")["official_coco"]["touching_pairs"] == shared_walls > 0

    # density strata come from the main ground truth
    make_density = load_script("make_density")
    make_density.main(["--data", "whu", "--split", "test", *common])
    make_density.main(["--data", "whu", "--split", "val_tune", *common])
    strata = read_json(paths.density_file("whu", "test", splits_dir))
    assert strata["contact_radius_px"] == 3 and sum(strata["main"]["sizes"].values()) == 27
