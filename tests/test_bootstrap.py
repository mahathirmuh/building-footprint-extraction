"""Bootstrap and comparisons (densepoly.eval.stats; docs/EVALUATION.md §4, EXPERIMENTS §8) on run_eval outputs of a
synthetic scene."""

import math

import numpy as np
import pytest

from densepoly.eval import coco_io, run_eval, stats
from tests.fixtures.coco_toy import make_results
from tests.fixtures.eval_scene import HOUSES, STRATA, predictions, shifted, write_scene

ALL_IDS = [1, 2, 3, 4, 5, 6]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """(records, tables) of five methods: the scene's predictions (a), the GT itself, a with the merge of image 6
    fixed, a without its predictions on image 2, and a plus the house it missed on image 1 (dense)."""
    with pytest.MonkeyPatch.context() as mp:
        gt, splits = write_scene(tmp_path_factory.mktemp("bootstrap"), mp)
        a = predictions()
        methods = {
            "a": a,
            "gt": coco_io.gt_as_predictions(gt),
            "fix_merge": [r for r in a if r["image_id"] != 6] + make_results({6: HOUSES[6]}),
            "no_image_2": [r for r in a if r["image_id"] != 2],
            "find_missed": a + make_results({1: [shifted(HOUSES[1][5], 1)]}, score=0.2),
        }
        records, tables = {}, {}
        for name, results in methods.items():
            records[name] = run_eval.evaluate("crowdai", "test", name, "ref", results, splits_dir=splits)
            tables[name] = stats.load_run("crowdai", name, "ref", "test")
        yield records, tables


def same(value, expected):
    if expected is None:
        return math.isnan(value)
    return value == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_weights_resample_with_replacement():
    weights = stats.bootstrap_weights(7, n_rep=50, seed=3)
    assert weights.shape == (50, 7) and (weights.sum(axis=1) == 7).all()
    assert (weights == stats.bootstrap_weights(7, n_rep=50, seed=3)).all()
    assert (weights != stats.bootstrap_weights(7, n_rep=50, seed=4)).any()
    strata = stats.stratified_weights([2, 5], n_rep=50)
    assert (strata[:, :2].sum(axis=1) == 2).all() and (strata[:, 2:].sum(axis=1) == 5).all()


@pytest.mark.parametrize("metric", [*stats.RATIO_METRICS, *stats.AP_METRICS])
def test_identity_weights_reproduce_the_summaries_of_run_eval(runs, metric):
    records, tables = runs
    for name in ("a", "fix_merge"):
        value = stats.metric_values(tables[name], metric, ALL_IDS, np.ones(6))[0]
        assert same(value, records[name]["overall"][metric])
        for stratum in ("padat", "sedang"):
            value = stats.metric_values(tables[name], metric, STRATA[stratum], np.ones(len(STRATA[stratum])))[0]
            assert same(value, records[name]["per_stratum"][stratum][metric])


@pytest.mark.parametrize("metric", ["AP", "AP_boundary", "merge_rate", "CIoU", "PoLiS_paired", "MTA"])
def test_a_method_against_itself_differs_by_nothing(runs, metric):
    _, tables = runs
    result = stats.compare([tables["a"]], [tables["a"]], metric, ALL_IDS, n_rep=100)
    assert result["delta"] == 0 and result["ci95"] == [0.0, 0.0]


def test_the_perfect_method_wins_on_ap_with_a_ci_above_zero(runs):
    records, tables = runs
    result = stats.compare([tables["a"]], [tables["gt"]], "AP", ALL_IDS, n_rep=200)
    assert result["delta"] == pytest.approx(records["gt"]["overall"]["AP"] - records["a"]["overall"]["AP"])
    assert result["delta"] > 0 and result["ci95"][0] > 0
    assert result == stats.compare([tables["a"]], [tables["gt"]], "AP", ALL_IDS, n_rep=200)  # fixed seed


def test_paired_metrics_only_use_the_gt_both_methods_matched(runs):
    _, tables = runs
    a, missing = tables["a"], tables["no_image_2"]
    result = stats.compare([a], [missing], "PoLiS_paired", ALL_IDS, n_rep=50)
    common = a.pairs[a.pairs["image_id"] != 2]
    assert result["a"] == [pytest.approx(common["polis"].mean())]
    assert result["delta"] == pytest.approx(0)  # same polygons on the GT matched by both
    assert stats.common_gt(a, missing, "PoLiS_paired") == set(common["gt_id"])


def test_seeds_are_compared_pairwise_and_averaged(runs):
    _, tables = runs
    result = stats.compare([tables["a"], tables["a"]], [tables["fix_merge"], tables["gt"]], "merge_rate", ALL_IDS)
    assert result["delta_per_seed"] == [pytest.approx(-2 / 13), pytest.approx(-2 / 13)]
    assert result["delta"] == pytest.approx(-2 / 13) and result["ci95"][1] <= 0


def test_stratified_difference_between_dense_and_sparse_tiles(runs):
    """Finding the missed house helps the dense tile only: Δpadat − Δjarang = 1/6, and with one dense tile and no
    change on the sparse ones every replicate gives that value."""
    _, tables = runs
    result = stats.compare_strata(
        [tables["a"]], [tables["find_missed"]], "recall_inst50", STRATA["padat"], STRATA["jarang"], n_rep=100
    )
    assert result["delta_dense"] == pytest.approx(1 / 6) and result["delta_sparse"] == 0
    assert result["difference"] == pytest.approx(1 / 6) and result["ci95"] == [pytest.approx(1 / 6)] * 2


def test_replicates_with_an_empty_denominator_are_counted_not_used(runs):
    """MTA_inst on images 3 and 5: a replicate of image 5 alone has no pair."""
    _, tables = runs
    result = stats.compare([tables["a"]], [tables["gt"]], "MTA_inst", [3, 5], n_rep=200)
    assert result["nan_replicates"] > 0 and result["ci95"] is not None


def test_ap_needs_the_cached_evalimgs_and_whole_draw_counts(runs):
    _, tables = runs
    without = stats.RunTables(tables["a"].per_image, tables["a"].pairs, None)
    with pytest.raises(ValueError, match="evalImgs"):
        stats.metric_values(without, "AP", ALL_IDS, np.ones(6))
    with pytest.raises(ValueError, match="whole numbers"):
        stats.metric_values(tables["a"], "AP", ALL_IDS, np.full(6, 0.5))


def test_seed_summary():
    assert stats.seed_summary([1.0, 2.0, 3.0]) == {"mean": 2.0, "sd": 1.0, "n": 3}
    assert stats.seed_summary([0.5]) == {"mean": 0.5, "sd": None, "n": 1}
