"""B0 predictions and the post-processing tuning on fake CrowdAI tiles (densepoly.infer, densepoly.tuning,
scripts/predict.py and scripts/tune_postproc.py; docs/EXPERIMENTS.md §4)."""

import importlib.util
import json
import shutil

import numpy as np
import pytest
from PIL import Image

from densepoly import infer, paths, tuning
from densepoly.classical.b0 import B0Params
from densepoly.config import load_config, postproc_file
from densepoly.eval import coco_io
from densepoly.postproc.polygonize import PolygonParams
from densepoly.priors.cache import load_stats, quantize, write_prior
from densepoly.priors.mbi import brightness, mbi
from densepoly.utils.io import read_json, write_json
from tests.fixtures.synthetic import make_scene, to_coco

SMALL_GRID = {"tie_ap": 0.2, "b0": {"t_min": [0.1, 0.3]},
              "polygon": {"rect_iou": [0.85, 0.93], "dp_k": [0.04], "snap_deg": [15]}}  # fmt: skip


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, paths.REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """crowdai/val_tune_sub of three synthetic scenes: GT, RGB files, and an MBI cache quantised with the frozen q of
    CrowdAI, as scripts/precompute_priors.py would write it. Returns (gt, a copy of configs/ without frozen tunings)."""
    monkeypatch.setenv("DENSEPOLY_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("DENSEPOLY_OUT", str(tmp_path / "outputs"))
    scenes = [make_scene(seed) for seed in range(3)]
    gt = to_coco(scenes)
    write_json(paths.gt_file("crowdai", "val_tune_sub"), gt)
    cfg = load_config("crowdai", "b0", tuned=False)
    q = load_stats("crowdai")["q"][0]
    for image, scene in zip(gt["images"], scenes, strict=True):
        folder = paths.image_dir("crowdai", "val_tune_sub")
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(scene.image).save(folder / image["file_name"])
        raw = mbi(brightness(scene.image), list(cfg.priors.mbi_scales_px))
        cached = quantize(raw[..., None], [q])[..., 0]
        write_prior(paths.prior_cache("crowdai", "val_tune_sub", image["id"]), np.dstack([cached, cached * 0]))
    configs = tmp_path / "configs"
    shutil.copytree(paths.configs_dir(), configs)
    shutil.rmtree(configs / "postproc" / "crowdai", ignore_errors=True)
    return gt, configs


def b0_job(gt):
    return infer.B0Job("crowdai", "val_tune_sub", B0Params(), PolygonParams("mar"), gt["categories"][0]["id"])


def test_grid_points_vary_the_last_axis_fastest():
    points = tuning.grid_points({"a": [1, 2], "b": [3, 4]})
    assert points == [{"a": 1, "b": 3}, {"a": 1, "b": 4}, {"a": 2, "b": 3}, {"a": 2, "b": 4}]


def test_the_grid_of_the_repo_has_36_points_per_stage():
    grid = tuning.load_grid()
    assert [len(tuning.grid_points(grid[k])) for k in ("instance", "polygon", "b0", "binary")] == [36, 36, 4, 3]


def test_choose_takes_the_best_ap_then_the_smallest_polis_within_the_tie():
    rows = [{"AP": 0.50, "PoLiS_paired": 1.0}, {"AP": 0.52, "PoLiS_paired": 0.9}, {"AP": 0.519, "PoLiS_paired": 0.5},
            {"AP": 0.40, "PoLiS_paired": 0.1}, {"AP": 0.5195, "PoLiS_paired": None}]  # fmt: skip
    assert tuning.choose(rows) == 1
    assert tuning.choose(rows, tie_ap=0.2) == 2  # 0.519 is within 0.2 AP points of 0.52
    assert tuning.choose(rows, tie_ap=0.05) == 1


def test_choose_takes_the_first_best_ap_when_no_row_has_pairs():
    rows = [{"AP": 0.0, "PoLiS_paired": None}, {"AP": 0.0, "PoLiS_paired": None}]
    assert tuning.choose(rows, tie_ap=0.2) == 0


def test_predictions_are_the_same_in_parallel_and_in_order(fake):
    gt, _ = fake
    ids = [image["id"] for image in gt["images"]]
    serial, timings = infer.predict_b0(b0_job(gt), ids, workers=1)
    parallel, _ = infer.predict_b0(b0_job(gt), ids, workers=2)
    assert serial == parallel and len(timings) == len(ids)
    assert [r["image_id"] for r in serial] == sorted(r["image_id"] for r in serial)
    coco_io.check_predictions(serial, gt, strict=True)


def test_tune_b0_scores_every_grid_point_and_picks_from_the_grid(fake):
    gt, _ = fake
    report = tuning.tune_b0(b0_job(gt), gt, SMALL_GRID, workers=1)
    assert [len(report["stage1"]), len(report["stage2"])] == [2, 2]
    assert report["chosen"]["b0"]["t_min"] in (0.1, 0.3) and report["chosen"]["rect_iou"] in (0.85, 0.93)
    assert report["score"] in [{"AP": r["AP"], "PoLiS_paired": r["PoLiS_paired"]} for r in report["stage2"]]
    again = tuning.tune_b0(b0_job(gt), gt, SMALL_GRID, workers=2)
    assert all(again[key] == report[key] for key in ("chosen", "stage1", "stage2"))


def test_predict_script_writes_predictions_meta_and_efficiency(fake):
    gt, configs = fake
    load_script("predict").main(["--data", "crowdai", "--exp", "b0", "--split", "val_tune_sub", "--workers", "1",
                                 "--time-sample", "2", "--configs-dir", str(configs)])  # fmt: skip
    pred_file = paths.pred_file("crowdai", "b0", "ref", "val_tune_sub")
    coco_io.check_predictions(read_json(pred_file), gt, strict=True)
    meta = read_json(pred_file.with_name("val_tune_sub.meta.json"))
    assert meta["efficiency"]["mbi_equal_to_cache"] == "2/2" and meta["tuned_file"] is None
    assert meta["postproc"]["b0"]["t_min"] == 0.15 and meta["images"] == 3


def test_predict_refuses_test_without_a_frozen_tuning(fake):
    _, configs = fake
    with pytest.raises(SystemExit):
        load_script("predict").main(["--data", "crowdai", "--exp", "b0", "--split", "test", "--configs-dir",
                                     str(configs)])  # fmt: skip


def test_tune_script_freezes_once_and_load_config_merges_it(fake, capsys):
    _, configs = fake
    (configs / "postproc" / "grid.yaml").write_text(json.dumps(SMALL_GRID), encoding="utf-8")  # JSON is YAML
    script = load_script("tune_postproc")
    base = ["--data", "crowdai", "--exp", "b0", "--workers", "1", "--configs-dir", str(configs)]
    script.main([*base, "--limit", "2", "--dry-run"])
    assert "projected_minutes_all_images" in capsys.readouterr().out
    assert not postproc_file("crowdai", "b0", configs).exists()
    script.main(base)
    frozen = load_config("crowdai", "b0", root=configs)
    document = read_yaml(postproc_file("crowdai", "b0", configs))
    assert frozen.postproc.b0.t_min == document["postproc"]["b0"]["t_min"]
    assert frozen.postproc.rect_iou == document["postproc"]["rect_iou"]
    assert document["tuning"]["split"] == "val_tune_sub" and len(document["tuning"]["stage2"]) == 2
    with pytest.raises(SystemExit):
        script.main(base)  # frozen: no overwrite without --force


def read_yaml(path):
    from omegaconf import OmegaConf

    return OmegaConf.to_container(OmegaConf.load(path))
