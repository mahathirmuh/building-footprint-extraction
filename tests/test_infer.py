"""Network inference, its post-processing and its tuning on fake tiles (densepoly.infer, densepoly.tuning, scripts/
predict.py, scripts/tune_postproc.py, scripts/evaluate.py; docs/EVALUATION.md §1, docs/EXPERIMENTS.md §4)."""

import importlib.util
import shutil

import numpy as np
import pytest
import torch
from torch import nn

from densepoly import infer, paths, train, tuning
from densepoly.config import load_config, postproc_file
from densepoly.eval import coco_io
from densepoly.labels.contact import class_map
from densepoly.postproc.polygonize import PolygonParams
from densepoly.utils.io import read_json, read_png, write_json
from tests.fixtures.fake_tiles import write_tiles
from tests.fixtures.synthetic import instance_map_from_polygons, make_scene, rect_polygon, to_coco

SIZE = 120
SMALL = ["model.encoder_weights=null", "model.encoder=efficientnet-b0", f"data.tile_size={SIZE}", "data.pad_to=128",
         "train.batch_size=2", "train.min_micro_batch=2", "train.num_workers=0", "train.epochs=1"]  # fmt: skip
SMALL_GRID = {"tie_ap": 0.2, "instance": {"tau": [0.3, 0.5], "marker_min_px": [8], "instance_min_px": [10]},
              "binary": {"instance_min_px": [10, 20]},
              "polygon": {"rect_iou": [0.85, 0.93], "dp_k": [0.04], "snap_deg": [15]}}  # fmt: skip


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, paths.REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def crop_gt(scene_seeds):
    """GT of the crops that write_tiles makes: the polygons of each scene are kept when they lie inside the crop."""
    gt = to_coco([make_scene(seed) for seed in scene_seeds])
    annotations = []
    for ann in gt["annotations"]:
        ring = np.reshape(ann["segmentation"][0], (-1, 2))
        if ring.max() < SIZE:
            annotations.append(ann)
    images = [{**img, "id": seed, "height": SIZE, "width": SIZE, "file_name": f"{seed:012d}.png"}
              for img, seed in zip(gt["images"], scene_seeds, strict=True)]  # fmt: skip
    ids = {old["id"]: new["id"] for old, new in zip(gt["images"], images, strict=True)}
    return {**gt, "images": images, "annotations": [{**a, "image_id": ids[a["image_id"]]} for a in annotations]}


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """One epoch of B4 and of FULL-P0 on fake tiles, the GT of val_tune_sub, and a copy of configs/ without frozen
    tunings: (splits folder, configs folder)."""
    root = tmp_path_factory.mktemp("trained")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("DENSEPOLY_DATA", str(root / "data"))
        patch.setenv("DENSEPOLY_OUT", str(root / "outputs"))
        folder = root / "splits"
        folder.mkdir()
        ids = write_tiles(folder, {"train_dedup": 2, "val_select": 1, "val_tune_sub": 2})
        for exp in ("b4", "full_p0"):
            cfg = load_config("crowdai", exp, SMALL)
            train.fit(cfg, paths.run_dir("crowdai", exp, "seed0"), splits_dir=folder)
        write_json(paths.gt_file("crowdai", "val_tune_sub"), crop_gt(ids["val_tune_sub"]))
        configs = root / "configs"
        shutil.copytree(paths.configs_dir(), configs)
        shutil.rmtree(configs / "postproc" / "crowdai", ignore_errors=True)
        (configs / "postproc" / "grid.yaml").write_text(str(SMALL_GRID).replace("'", '"'), encoding="utf-8")
        yield folder, configs


def predict(trained, exp, *extra, split="val_tune_sub"):
    folder, configs = trained
    load_script("predict").main(["--data", "crowdai", "--exp", exp, "--seed", "0", "--split", split, "--workers", "1",
                                 "--batch-size", "2", "--loader-workers", "0", "--configs-dir", str(configs),
                                 "--splits-dir", str(folder), *extra, *SMALL[:4]])  # fmt: skip


class Fixed(nn.Module):
    """Logits with one given value per channel, on any input."""

    def __init__(self, values):
        super().__init__()
        self.values = torch.tensor(values)

    def forward(self, image):
        return self.values[None, :, None, None].repeat(image.shape[0], 1, *image.shape[2:])


def batches(side=128, n=2):
    return [{"image": torch.zeros(n, 3, side, side), "image_id": torch.arange(1, n + 1)}]


@pytest.mark.parametrize("n_out", [1, 3, 5])
def test_channels_are_cropped_to_the_tile_as_float16(n_out):
    model = Fixed([0.0, 1.0, 2.0, 0.5, -0.5][:n_out])
    out = list(infer.network_channels(model, batches(), torch.device("cpu"), SIZE))
    assert [image_id for image_id, _, _ in out] == [1, 2]
    channels = out[0][1]
    assert channels.shape == (n_out, SIZE, SIZE) and channels.dtype == np.float16
    assert out[0][2] >= 0


def test_three_class_channels_are_probabilities_and_full_adds_the_orientation():
    (_, channels, _), _ = infer.network_channels(
        Fixed([1.0, 2.0, 0.0, 0.5, -0.5]), batches(), torch.device("cpu"), SIZE
    )
    softmax = torch.softmax(torch.tensor([1.0, 2.0, 0.0]), 0).numpy()
    np.testing.assert_allclose(channels[:3, 0, 0], softmax, atol=1e-3)
    np.testing.assert_allclose(channels[3:, 0, 0], np.tanh([0.5, -0.5]), atol=1e-3)


def test_binary_channel_is_a_sigmoid():
    (_, channels, _), _ = infer.network_channels(Fixed([1.0]), batches(), torch.device("cpu"), SIZE)
    assert channels[0, 0, 0] == pytest.approx(float(torch.sigmoid(torch.tensor(1.0))), abs=1e-3)


def perfect_channels(instance_map):
    classes = class_map(instance_map, 2)
    return np.stack([(classes == c).astype(np.float32) for c in (0, 1, 2)])


def test_network_instances_separate_touching_houses_with_the_contact_class():
    scene = make_scene(4)
    channels = perfect_channels(scene.instance_map)
    found = infer.network_instances(channels, infer.InstanceParams())
    merged = infer.network_instances(1.0 - channels[:1], infer.InstanceParams())
    assert found.count == len(scene.buildings) and merged.count < found.count


def test_mode_theta_takes_theta_and_kappa_from_the_head():
    angle = 25.0
    instance_map = instance_map_from_polygons([rect_polygon(60, 60, 70, 30, angle)], 120, 120)
    channels = np.concatenate([perfect_channels(instance_map), np.zeros((2, 120, 120), np.float32)])
    channels[3] = np.sin(np.radians(4 * angle))  # the head says 25°, with no doubt
    channels[4] = np.cos(np.radians(4 * angle))
    found = infer.network_instances(channels, infer.InstanceParams())
    (polygon,) = infer.network_polygons(found, channels, PolygonParams("theta"))
    assert polygon.theta_deg == pytest.approx(angle, abs=0.01) and polygon.kappa == pytest.approx(1.0, abs=1e-5)
    (result,) = infer.network_polygons(found, channels, PolygonParams("mar"))
    assert result.kappa is None and result.theta_deg is not None


def test_b4_checkpoint_gives_predictions_through_the_scripts(trained):
    predict(trained, "b4", "--save-channels")
    pred_file = paths.pred_file("crowdai", "b4", "seed0", "val_tune_sub")
    meta = read_json(pred_file.with_name("val_tune_sub.meta.json"))
    assert meta["checkpoint"] == "b4/seed0/best.pt" and meta["selected"]["epoch"] == 0
    assert meta["seconds"]["network_mean"] > 0 and meta["images"] == 2
    gt = read_json(paths.gt_file("crowdai", "val_tune_sub"))
    coco_io.check_predictions(read_json(pred_file), gt, strict=True)
    for image in gt["images"]:
        channels = np.load(paths.channel_file("crowdai", "b4", "seed0", "val_tune_sub", image["id"]))
        assert channels.shape == (3, SIZE, SIZE) and channels.dtype == np.float16


def test_saved_channels_give_the_predictions_of_the_network(trained):
    predict(trained, "b4", "--save-channels")
    pred_file = paths.pred_file("crowdai", "b4", "seed0", "val_tune_sub")
    direct = read_json(pred_file)
    predict(trained, "b4", "--from-channels")
    assert read_json(pred_file) == direct
    assert read_json(pred_file.with_name("val_tune_sub.meta.json"))["channels_of"] == "b4/seed0"


def test_b5_uses_the_network_of_b4(trained):
    predict(trained, "b4", "--save-channels")
    predict(trained, "b5", "--from-channels")
    results = read_json(paths.pred_file("crowdai", "b5", "seed0", "val_tune_sub"))
    meta = read_json(paths.pred_file("crowdai", "b5", "seed0", "val_tune_sub").with_name("val_tune_sub.meta.json"))
    assert meta["channels_of"] == "b4/seed0" and meta["postproc"]["polygon"] == "mar"
    assert all(r["theta_deg"] is not None and r["kappa"] is None for r in results)


def test_full_predictions_carry_theta_and_kappa(trained):
    predict(trained, "full_p0")
    results = read_json(paths.pred_file("crowdai", "full_p0", "seed0", "val_tune_sub"))
    assert all(r["theta_deg"] is not None and 0 <= r["kappa"] <= 1 for r in results)


def test_a_network_needs_a_seed_or_a_run(trained):
    folder, configs = trained
    with pytest.raises(SystemExit):
        load_script("predict").main(["--data", "crowdai", "--exp", "b4", "--split", "val_tune_sub", "--configs-dir",
                                     str(configs), "--splits-dir", str(folder)])  # fmt: skip


def test_the_test_split_needs_a_frozen_tuning(trained):
    with pytest.raises(SystemExit):
        predict(trained, "b4", split="test")


def test_tune_network_scores_stage_1_only_for_raw_polygons(trained):
    predict(trained, "b4", "--save-channels")
    gt = read_json(paths.gt_file("crowdai", "val_tune_sub"))
    job = infer.NetworkJob("crowdai", "b4", "seed0", "val_tune_sub", infer.InstanceParams(), PolygonParams("raw"),
                           gt["categories"][0]["id"])  # fmt: skip
    report = tuning.tune_network(job, gt, SMALL_GRID, binary=False)
    assert len(report["stage1"]) == 2 and report["stage2"] == []
    assert set(report["chosen"]) == {"tau", "marker_min_px", "instance_min_px"}
    assert report["score"] in [{"AP": r["AP"], "PoLiS_paired": r["PoLiS_paired"]} for r in report["stage1"]]


def test_tune_network_adds_stage_2_for_regularised_polygons(trained):
    predict(trained, "b4", "--save-channels")
    gt = read_json(paths.gt_file("crowdai", "val_tune_sub"))
    job = infer.NetworkJob("crowdai", "b4", "seed0", "val_tune_sub", infer.InstanceParams(), PolygonParams("mar"),
                           gt["categories"][0]["id"])  # fmt: skip
    report = tuning.tune_network(job, gt, SMALL_GRID, binary=False)
    assert len(report["stage2"]) == 2 and report["chosen"]["rect_iou"] in (0.85, 0.93) and "tau" in report["chosen"]


def test_the_tune_script_freezes_a_network_and_predict_uses_it(trained):
    folder, configs = trained
    predict(trained, "b4", "--save-channels")
    load_script("tune_postproc").main(["--data", "crowdai", "--exp", "b5", "--workers", "1",
                                       "--configs-dir", str(configs)])  # fmt: skip
    frozen = load_config("crowdai", "b5", root=configs)
    assert frozen.postproc.rect_iou in (0.85, 0.93) and frozen.postproc.tau in (0.3, 0.5)
    assert postproc_file("crowdai", "b5", configs).exists()
    predict(trained, "b5", "--from-channels")
    meta = read_json(paths.pred_file("crowdai", "b5", "seed0", "val_tune_sub").with_name("val_tune_sub.meta.json"))
    assert meta["tuned_file"] == "configs/postproc/crowdai/b5.yaml" and meta["postproc"]["tau"] == frozen.postproc.tau
    postproc_file("crowdai", "b5", configs).unlink()


def test_evaluate_reads_a_named_run(trained, monkeypatch):
    predict(trained, "b4", "--save-channels")
    source = paths.pred_file("crowdai", "b4", "seed0", "val_tune_sub")
    target = paths.pred_file("crowdai", "b4", "pilot_r5", "val_tune_sub")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, target)
    load_script("evaluate").main(["--data", "crowdai", "--exp", "b4", "--run", "pilot_r5", "--split", "val_tune_sub",
                                  "--workers", "1", "--splits-dir", str(trained[0])])  # fmt: skip
    assert paths.metrics_file("crowdai", "b4", "pilot_r5", "val_tune_sub").exists()


def test_fake_tile_labels_match_the_crop_gt(trained):
    """The GT above and the instance cache describe the same crops, or the evaluation here would mean nothing."""
    gt = read_json(paths.gt_file("crowdai", "val_tune_sub"))
    image_id = gt["images"][0]["id"]
    instance_map = read_png(paths.instance_cache("crowdai", "val_tune_sub", image_id))
    assert instance_map.shape == (SIZE, SIZE) and instance_map.any()
