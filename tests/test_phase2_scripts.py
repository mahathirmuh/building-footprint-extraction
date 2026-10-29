"""End-to-end run of the Phase 2 scripts on the fake downloads (tests/fixtures/fake_crowdai.py, fake_whu.py): prior
benchmark, statistics and cache, label statistics, and the QA figures.

These tests check the plumbing (units, scaling, file layout, resuming, figures). The priors themselves are replaced
by a fast stand-in, since the real MBI takes ±13 s per 1024² tile; MBI and the gap map are tested on synthetic scenes
in tests/test_priors.py."""

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from densepoly import paths
from densepoly.config import load_config
from densepoly.priors import cache
from densepoly.priors.mbi import brightness
from densepoly.utils.io import read_json
from tests.fixtures.fake_crowdai import make_fake_crowdai
from tests.fixtures.fake_whu import make_fake_whu


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, paths.REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fast_priors(image, priors):
    """Stand-in for cache.compute_priors with the same shape and dtype: the brightness and its square."""
    b = brightness(image)
    return np.stack([b, b**2], axis=-1)


@pytest.fixture(scope="module")
def phase1(tmp_path_factory):
    """Fake downloads with the Phase 1 outputs the Phase 2 scripts read; prior statistics go to a temporary folder."""
    root = tmp_path_factory.mktemp("phase2")
    make_fake_crowdai(root / "data")
    make_fake_whu(root / "data")
    patch = pytest.MonkeyPatch()
    patch.setenv("DENSEPOLY_DATA", str(root / "data"))
    patch.setenv("DENSEPOLY_OUT", str(root / "outputs"))
    patch.setattr(cache, "stats_file", lambda dataset: root / f"{dataset}_prior_stats.json")
    patch.setattr(cache, "compute_priors", fast_priors)  # workers=1 everywhere, so no process re-imports the module
    common = ["--splits-dir", str(root / "splits")]
    load_script("dedup_crowdai").main(["--data", "crowdai", "--workers", "1", *common])
    load_script("prepare_crowdai").main(["--cache", "train_dedup", "val_tune", "test", "val_clean", *common])
    load_script("prepare_whu_pix2poly").main(["--workers", "1", *common])
    load_script("prepare_whu_pix2poly").main(["--cache", "train_dedup", "val_tune", "test", *common])
    for data in ("crowdai", "whu"):
        for split in ("test", "val_tune"):
            load_script("make_density").main(["--data", data, "--split", split, *common])
    yield SimpleNamespace(root=root, splits=root / "splits", common=common)
    patch.undo()


def test_prior_benchmark_statistics_and_cache(phase1):
    script = load_script("precompute_priors")
    script.main(["--data", "crowdai", "--benchmark", "2", "--workers", "1", *phase1.common])
    assert not (phase1.root / "crowdai_prior_stats.json").exists()  # the benchmark writes nothing

    script.main(["--data", "crowdai", "--stats", "--workers", "1", *phase1.common])
    stats = read_json(phase1.root / "crowdai_prior_stats.json")
    assert stats["units"] == 8 and len(stats["q"]) == 2 and min(stats["q"]) > 0  # all 8 fake train_dedup images

    script.main(["--data", "crowdai", "--split", "val_tune", "--workers", "1", *phase1.common])
    unit = cache.units("crowdai", "val_tune", phase1.splits)[0]
    priors = OmegaConf.to_container(load_config("crowdai", "full_p2").priors)
    expected = cache.quantize(cache.compute_priors(cache.read_rgb(unit.image), priors), stats["q"])
    np.testing.assert_array_equal(cache.read_prior(paths.prior_cache("crowdai", "val_tune", unit.key)), expected)
    script.main(["--data", "crowdai", "--split", "val_tune", "--workers", "1", *phase1.common])
    timings = read_json(phase1.splits / "crowdai_prior_cache.json")["val_tune"]
    assert (timings["units"], timings["computed_now"], timings["already_cached"]) == (1, 0, 1)  # resumed

    # WHU priors are computed on the official 1024² tiles of the split's patches
    script.main(["--data", "whu", "--stats", "--workers", "1", *phase1.common])
    assert read_json(phase1.root / "whu_prior_stats.json")["units"] == 2  # tile 10002 has no patch
    script.main(["--data", "whu", "--split", "test", "--workers", "1", *phase1.common])
    assert [u.key for u in cache.units("whu", "test", phase1.splits)] == ["3000"]  # part 300 only
    assert cache.read_prior(paths.prior_cache("whu", "test", "3000")).shape == (1024, 1024, 2)


def test_label_statistics(phase1):
    load_script("label_stats").main(["--data", "crowdai", "--split", "train_dedup", "--workers", "1", *phase1.common])
    stats = read_json(phase1.splits / "crowdai_label_stats.json")["train_dedup"]
    assert stats["tiles"] == 8 and stats["contact_radius_px"] == 2
    assert 0 < stats["contact_share_of_building_px"] < 0.2  # the touching rows of the synthetic scenes
    assert stats["theta_valid_share_of_building_px"] == 1.0  # every synthetic house is above A_θ


@pytest.mark.parametrize("dataset", ["crowdai", "whu"])
def test_qa_figures(phase1, dataset):
    load_script("figures").main(["--data", dataset, "--qa-labels", "--qa-gap-mode", *phase1.common])
    for name in ("labels", "gap_mode"):
        assert (phase1.root / "outputs" / "figures" / "qa" / f"{dataset}_{name}.png").is_file()
