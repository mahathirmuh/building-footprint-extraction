"""Config layering (base <- data <- exp <- CLI) and the scenario matrix of docs/EXPERIMENTS.md §2."""

from itertools import pairwise

import pytest
from omegaconf import OmegaConf

from densepoly import paths
from densepoly.config import load_config, save_config

# docs/EXPERIMENTS.md §2: arch, output channels, input priors, loss, edge weight, instances, polygons, trained
MATRIX = {
    "b0": ("classical", None, [], "none", False, "classical", "mar", False),
    "b1": ("unet", 1, [], "bce_dice", False, "cc", "raw", True),
    "b2": ("unetplusplus", 1, [], "bce_dice", False, "cc", "raw", True),
    "b3": ("unetplusplus", 1, [], "bce_dice", True, "cc", "raw", True),
    "b4": ("unetplusplus", 3, [], "ce_dice", True, "watershed", "raw", True),
    "b5": ("unetplusplus", 3, [], "none", False, "watershed", "mar", False),
    "b5p": ("unetplusplus", 5, [], "none", False, "watershed", "mar", False),
    "full_p0": ("unetplusplus", 5, [], "ce_dice_theta", True, "watershed", "theta", True),
    "full_p1": ("unetplusplus", 5, ["mbi"], "ce_dice_theta", True, "watershed", "theta", True),
    "full_p2": ("unetplusplus", 5, ["mbi", "gap"], "ce_dice_theta", True, "watershed", "theta", True),
    "full_y": ("unetplusplus", 5, None, "none", False, "watershed_yolo", "theta", False),
}


def test_experiment_files_match_the_matrix():
    assert sorted(p.stem for p in (paths.configs_dir() / "exp").glob("*.yaml")) == sorted(MATRIX)


@pytest.mark.parametrize("dataset", ["crowdai", "whu"])
@pytest.mark.parametrize("exp", sorted(MATRIX))
def test_every_experiment_loads_for_every_dataset(dataset, exp):
    cfg = load_config(dataset, exp)
    assert (cfg.data.name, cfg.exp) == (dataset, exp)
    assert cfg.data.pad_to % 32 == 0 and cfg.data.pad_to >= cfg.data.tile_size


@pytest.mark.parametrize("exp", sorted(MATRIX))
def test_configs_encode_the_scenario_matrix(exp):
    cfg = load_config("crowdai", exp)
    priors = None if cfg.input.priors is None else list(cfg.input.priors)
    row = (
        cfg.model.arch,
        cfg.model.classes,
        priors,
        cfg.loss.type,
        cfg.loss.edge_weight,
        cfg.postproc.instance,
        cfg.postproc.polygon,
        cfg.train.enabled,
    )
    assert row == MATRIX[exp]


@pytest.mark.parametrize("exp", ["b5", "b5p"])
def test_inference_only_configs_mirror_their_source(exp):
    cfg = load_config("crowdai", exp)
    src = load_config("crowdai", cfg.model.source_exp)
    assert src.train.enabled and not cfg.train.enabled
    assert (cfg.model.arch, cfg.model.classes) == (src.model.arch, src.model.classes)
    assert list(cfg.input.priors) == list(src.input.priors)


def test_whu_pixel_parameters_follow_the_gsd_rule():
    """docs/METHOD.md §10: WHU 0.2 m = CrowdAI 0.3 m values × 1.5 for lengths and × 2.25 for areas."""
    crowdai, whu = load_config("crowdai", "b4"), load_config("whu", "b4")
    assert whu.labels.contact_radius_px == round(crowdai.labels.contact_radius_px * 1.5)
    assert whu.labels.edge_weight_sigma_px == crowdai.labels.edge_weight_sigma_px * 1.5
    assert whu.labels.theta_min_area_px == round(crowdai.labels.theta_min_area_px * 2.25)
    gap_se = whu.priors.gap_se_length_px
    assert gap_se % 2 == 1 and abs(gap_se - crowdai.priors.gap_se_length_px * 1.5) <= 1
    for cfg, step in ((crowdai, 10), (whu, 15)):
        scales = list(cfg.priors.mbi_scales_px)
        assert len(scales) == 8 and all(b - a == step for a, b in pairwise(scales))


def write_tree(root, base, data, exp):
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "exp").mkdir(exist_ok=True)
    (root / "base.yaml").write_text(base, encoding="utf-8")
    (root / "data" / "toy.yaml").write_text(data, encoding="utf-8")
    (root / "exp" / "e1.yaml").write_text(exp, encoding="utf-8")
    return root


BASE = "exp: ???\ndata:\n  name: ???\nvalue: 1\n"


def test_later_layers_win(tmp_path):
    root = write_tree(tmp_path, BASE, "data: {name: toy}\nvalue: 2\n", "exp: e1\nvalue: 3\n")
    assert load_config("toy", "e1", root=root).value == 3
    assert load_config("toy", "e1", ["value=4"], root=root).value == 4


def test_unknown_key_in_a_yaml_layer_raises(tmp_path):
    root = write_tree(tmp_path, BASE, "data: {name: toy}\n", "exp: e1\nvalu: 3\n")
    with pytest.raises(ValueError, match="valu"):
        load_config("toy", "e1", root=root)


def test_missing_mandatory_value_raises(tmp_path):
    root = write_tree(tmp_path, BASE + "needed: ???\n", "data: {name: toy}\n", "exp: e1\n")
    with pytest.raises(ValueError, match="needed"):
        load_config("toy", "e1", root=root)


def test_names_must_match_the_files(tmp_path):
    root = write_tree(tmp_path, BASE, "data: {name: toy}\n", "exp: other\n")
    with pytest.raises(ValueError, match="do not match"):
        load_config("toy", "e1", root=root)


def test_cli_overrides():
    cfg = load_config("crowdai", "b4", ["train.lr=1e-3", "seed=2"])
    assert cfg.train.lr == pytest.approx(1e-3) and cfg.seed == 2
    with pytest.raises(ValueError, match="lrr"):
        load_config("crowdai", "b4", ["train.lrr=1e-3"])
    with pytest.raises(ValueError, match="key=value"):
        load_config("crowdai", "b4", ["train.lr"])


def test_saved_config_round_trips(tmp_path):
    cfg = load_config("crowdai", "full_p2", ["seed=1"])
    path = save_config(cfg, tmp_path / "run")
    assert path.name == "config.yaml"
    assert OmegaConf.load(path) == cfg
