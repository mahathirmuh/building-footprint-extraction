"""Path helpers resolve data and outputs from the environment and follow docs/DATA.md §1, docs/EVALUATION.md §1."""

import pytest

from densepoly import paths


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("DENSEPOLY_DATA", raising=False)
    monkeypatch.delenv("DENSEPOLY_OUT", raising=False)


def test_repo_root_is_the_project_root():
    assert (paths.REPO_ROOT / "pyproject.toml").is_file()
    assert (paths.REPO_ROOT / "CLAUDE.md").is_file()


def test_defaults_live_inside_the_repo(clean_env):
    assert paths.data_root() == paths.REPO_ROOT / "data"
    assert paths.out_root() == paths.REPO_ROOT / "outputs"
    assert paths.configs_dir() == paths.REPO_ROOT / "configs"


def test_environment_overrides_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("DENSEPOLY_DATA", str(tmp_path / "d"))
    monkeypatch.setenv("DENSEPOLY_OUT", str(tmp_path / "o"))
    data, out = tmp_path.resolve() / "d", tmp_path.resolve() / "o"
    assert paths.data_root() == data
    assert paths.instance_cache("crowdai", "test", 7) == data / "instances/crowdai/test/7.png"
    assert paths.prior_cache("whu", "train_dedup", "a") == data / "priors/whu/train_dedup/a.png"
    assert paths.run_dir("crowdai", "full_p2", "seed0") == out / "runs/crowdai/full_p2/seed0"
    assert paths.pred_file("crowdai", "b4", "seed0", "test") == out / "preds/crowdai/b4/seed0/test.json"
    assert paths.metrics_file("whu", "b0", "ref", "val_tune") == out / "metrics/whu/b0/ref/val_tune.json"
    assert paths.gt_file("crowdai", "val_full") == out / "gt/crowdai/val_full.json"


def test_unknown_names_are_rejected():
    with pytest.raises(ValueError, match="dataset"):
        paths.pred_file("CrowdAI", "b4", "seed0", "test")
    with pytest.raises(ValueError, match="split"):
        paths.gt_file("crowdai", "val")


def test_split_and_density_file_names_follow_docs_data_section_6(tmp_path):
    assert paths.split_list("crowdai", "test").name == "crowdai_clean_test.txt"
    assert paths.split_list("crowdai", "val_tune_sub").name == "crowdai_val_tune_sub.txt"
    assert paths.split_list("whu", "test", root=tmp_path) == tmp_path / "whu_test.txt"
    assert paths.density_file("crowdai", "test") == paths.REPO_ROOT / "splits" / "density_crowdai_test.json"


def test_split_sources_follow_the_data_layout(monkeypatch, tmp_path):
    monkeypatch.setenv("DENSEPOLY_DATA", str(tmp_path))
    data = tmp_path.resolve()
    crowdai = {
        s: paths.source_folder("crowdai", s) for s in ("train_dedup", "val_tune", "val_select", "test", "val_full")
    }
    assert crowdai == {
        "train_dedup": "train",
        "val_tune": "train",
        "val_select": "train",
        "test": "val",
        "val_full": "val",
    }
    whu = {s: paths.source_folder("whu", s) for s in ("train_dedup", "val_tune", "val_tune_sub", "test")}
    assert whu == {"train_dedup": "train", "val_tune": "val", "val_tune_sub": "val", "test": "test"}
    assert paths.annotation_file("crowdai", "test") == data / "crowdai/val/annotation.json"
    assert paths.annotation_file("whu", "val_select") == data / "whu/pix2poly_224/val/annotation.json"
    assert paths.image_dir("whu", "train_dedup") == data / "whu/pix2poly_224/train/images"


def test_run_name():
    assert paths.run_name(0) == "seed0"
    assert paths.run_name(None) == "ref"
