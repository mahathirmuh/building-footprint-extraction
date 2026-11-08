"""Result tables and the decisions of docs/EXPERIMENTS.md §8 (densepoly.eval.tables) on run_eval outputs of a synthetic
scene, and the decision rules on numbers whose answer is known."""

import importlib.util
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from densepoly import paths
from densepoly.eval import coco_io, run_eval, stats, tables
from densepoly.utils.io import read_json, write_json
from tests.fixtures.coco_toy import make_results
from tests.fixtures.eval_scene import HOUSES, predictions, write_scene

SEED_RUNS = ("seed0", "seed1", "seed2")


def metrics_root():
    return paths.out_root() / "metrics" / "crowdai"


def install(**methods):
    """Make each experiment (keyword) a three-seed copy of an evaluated method of the scene (value)."""
    for exp, source in methods.items():
        shutil.rmtree(metrics_root() / exp, ignore_errors=True)
        for run in SEED_RUNS:
            shutil.copytree(metrics_root() / "scene" / source, metrics_root() / exp / run)


def make_dense(splits, image_ids):
    write_json(paths.density_file("crowdai", "test", splits), {"main": {"strata": {"padat": image_ids}}})


@pytest.fixture(scope="module")
def splits(tmp_path_factory):
    """Splits folder of a scene evaluated three ways: its predictions (a), the GT itself (gt), and a with the merge of
    image 6 fixed (fix_merge); plus b0 as a single seedless run."""
    with pytest.MonkeyPatch.context() as mp:
        gt, folder = write_scene(tmp_path_factory.mktemp("tables"), mp)
        a = predictions()
        methods = {
            "a": a,
            "gt": coco_io.gt_as_predictions(gt),
            "fix_merge": [r for r in a if r["image_id"] != 6] + make_results({6: HOUSES[6]}),
        }
        for name, results in methods.items():
            run_eval.evaluate("crowdai", "test", "scene", name, results, splits_dir=folder)
        run_eval.evaluate("crowdai", "test", "b0", "ref", a, splits_dir=folder)
        yield folder


def test_cell_shows_percent_a_decimal_comma_and_a_dash_for_no_value():
    assert tables.cell("AP", 0.6324) == "63,24"
    assert tables.cell("PoLiS_paired", 1.635) == "1,635"
    assert tables.cell("MTA_inst", 37.384) == "37,38"
    assert tables.cell("merge_rate", -0.0123, signed=True) == "-1,23"
    assert tables.cell("AP", 0.01, signed=True) == "+1,00"
    assert tables.cell("AP", None) == "—" and tables.cell("AP", float("nan")) == "—"


def outcome(per_seed, ci):
    return {"delta": float(np.mean(per_seed)), "delta_per_seed": list(per_seed), "ci95": ci}


@pytest.mark.parametrize(
    ("per_seed", "ci", "passed"),
    [
        ([-0.1, -0.2, -0.3], [-0.3, -0.05], True),
        ([-0.1, -0.2, -0.3], [-0.3, 0.01], False),  # the CI contains 0
        ([-0.1, -0.2, -0.3], [-0.3, 0.0], False),  # the CI touches 0
        ([-0.1, -0.2, 0.01], [-0.3, -0.05], False),  # one seed goes the other way
        ([-0.1, -0.2, 0.0], [-0.3, -0.05], False),  # one seed does not move
        ([-0.1, -0.2, -0.3], None, False),  # no valid replicate
        ([0.1, 0.2, 0.3], [0.05, 0.3], False),  # clearly worse
    ],
)
def test_improved_needs_the_ci_below_zero_and_every_seed_lower(per_seed, ci, passed):
    assert tables.improved(outcome(per_seed, ci))["passed"] is passed


@pytest.mark.parametrize(("delta", "passed"), [(0.01, True), (0.0, True), (-0.005, True), (-0.0051, False), (-0.1, False)])
def test_ap_may_fall_by_half_a_point_at_most(delta, passed):
    assert tables.ap_kept({"delta": delta})["passed"] is passed


def test_decide_n1_needs_both_geometry_metrics_and_the_ap():
    good, bad = outcome([-1, -1, -1], [-1.5, -0.5]), outcome([-1, -1, 1], [-1.5, 0.5])
    ap_ok, ap_low = {"delta": -0.001}, {"delta": -0.01}
    assert tables.decide_n1({"PoLiS_paired": good, "MTA_inst": good, "AP": ap_ok})["supported"] is True
    assert tables.decide_n1({"PoLiS_paired": good, "MTA_inst": bad, "AP": ap_ok})["supported"] is False
    assert tables.decide_n1({"PoLiS_paired": bad, "MTA_inst": good, "AP": ap_ok})["supported"] is False
    assert tables.decide_n1({"PoLiS_paired": good, "MTA_inst": good, "AP": ap_low})["supported"] is False


def test_decide_n3_needs_the_merge_rate_and_the_ap():
    good, bad = outcome([-1, -1, -1], [-1.5, -0.5]), outcome([-1, -1, -1], [-1.5, 0.5])
    assert tables.decide_n3({"merge_rate": good, "AP": {"delta": 0.0}})["supported"] is True
    assert tables.decide_n3({"merge_rate": bad, "AP": {"delta": 0.0}})["supported"] is False
    assert tables.decide_n3({"merge_rate": good, "AP": {"delta": -0.006}})["supported"] is False


@pytest.mark.parametrize(
    ("p0", "p2", "choice", "reason"),
    [
        ([0.62, 0.63, 0.64], [0.64, 0.65, 0.66], "full_p2", "higher_ap"),  # P2 ahead by 2 points
        ([0.64, 0.65, 0.66], [0.62, 0.63, 0.64], "full_p0", "higher_ap"),  # P0 ahead by 2 points
        ([0.63, 0.63, 0.63], [0.631, 0.631, 0.631], "full_p0", "tie"),  # P2 ahead by 0.1 point: fewer channels
        ([0.631, 0.631, 0.631], [0.63, 0.63, 0.63], "full_p0", "tie"),
        ([0.60, 0.63, 0.66], [0.66, 0.63, 0.60], "full_p0", "tie"),  # seeds differ, the means do not
    ],
)
def test_full_final_takes_the_higher_mean_ap_and_prefers_fewer_channels_on_a_tie(p0, p2, choice, reason):
    decision = tables.choose_full_final({"full_p0": p0, "full_p2": p2})
    assert (decision["choice"], decision["reason"]) == (choice, reason)
    assert decision["difference"] == pytest.approx(np.mean(p2) - np.mean(p0))


def test_full_final_table_reads_the_ap_of_val_select(splits):
    for exp, ap in (("full_p0", [0.60, 0.61, 0.62]), ("full_p2", [0.63, 0.64, 0.65])):
        for run, value in zip(SEED_RUNS, ap, strict=True):
            write_json(paths.metrics_file("crowdai", exp, run, "val_select"), {"overall": {"AP": value}, "git": "abc"})
    payload, lines = tables.full_final("crowdai")
    assert payload["decision"]["choice"] == "full_p2"
    assert payload["ap_per_seed"]["full_p2"] == [0.63, 0.64, 0.65]
    assert [entry["git"] for entry in payload["sources"]["full_p0"]] == ["abc"] * 3
    assert any("FULL final: FULL-P2" in line for line in lines)


def test_n1_when_the_head_is_perfect_on_the_dense_tiles(splits):
    """Image 4 holds one house, predicted 5° off by a and exactly by the GT."""
    install(b5="a", b5p="a", full_p0="gt")
    make_dense(splits, [4])
    payload, lines = tables.n1("crowdai", n_rep=30, splits_dir=splits)
    total = payload["steps"]["b5->full_p0"]
    assert total["PoLiS_paired"]["delta"] < 0 and total["MTA_inst"]["delta"] < 0
    assert total["AP"]["delta"] > 0
    assert payload["decision"]["supported"] is True
    assert payload["steps"]["b5->b5p"]["PoLiS_paired"]["delta"] == 0  # same predictions
    full = payload["methods"]["full_p0"]
    assert full["PoLiS_paired"]["summary"]["mean"] == pytest.approx(0, abs=1e-9)
    assert full["PoLiS_paired"]["summary"]["sd"] == pytest.approx(0, abs=1e-9)
    runs = tables.load_runs("crowdai", ("b5", "full_p0"), "test")
    assert total["AP"] == stats.compare(runs["b5"], runs["full_p0"], "AP", [4], 30, tables.BOOTSTRAP_SEED)
    assert payload["n_images"] == 1 and any("**N1 didukung**" in line for line in lines)


def test_n1_is_not_supported_when_nothing_changes(splits):
    install(b5="a", b5p="a", full_p0="a")
    make_dense(splits, [4])
    payload, lines = tables.n1("crowdai", n_rep=30, splits_dir=splits)
    assert payload["decision"]["supported"] is False
    assert payload["decision"]["checks"]["PoLiS_paired"]["ci_below_zero"] is False
    assert payload["decision"]["checks"]["AP"]["passed"] is True
    assert any("**N1 tidak didukung**" in line for line in lines)


def test_n3_when_the_prior_fixes_a_merge_on_the_dense_tiles(splits):
    """Image 6: one polygon over both houses in a, both found separately in fix_merge."""
    install(full_p0="a", full_p2="fix_merge")
    make_dense(splits, [6])
    payload, lines = tables.n3("crowdai", n_rep=30, splits_dir=splits)
    step = payload["steps"]["full_p0->full_p2"]
    assert step["merge_rate"]["delta"] == pytest.approx(-1.0)
    assert step["AP"]["delta"] > 0
    assert payload["decision"]["supported"] is True
    assert payload["methods"]["full_p2"]["merge_rate"]["summary"]["mean"] == pytest.approx(0)
    assert any("**N3 didukung**" in line for line in lines)


def test_values_of_the_methods_use_the_gt_that_all_of_them_matched():
    def run(gt_ids, polis, theta):
        pairs = pd.DataFrame(
            {
                "image_id": [1] * len(gt_ids),
                "gt_id": gt_ids,
                "polis": polis,
                "mta": [1.0] * len(gt_ids),
                "theta_err": theta,
            }
        )
        return stats.RunTables(pd.DataFrame({"image_id": [1]}).set_index("image_id", drop=False), pairs)

    a = run([10, 11, 12], [1.0, 2.0, 9.0], [1.0, 3.0, 40.0])
    b = run([10, 11], [0.0, 1.0], [float("nan"), 5.0])
    values = tables.method_values({"a": [a], "b": [b]}, ["PoLiS_paired"], [1], theta=True)
    assert values["a"]["PoLiS_paired"]["per_seed"] == [pytest.approx(1.5)]  # GT 12 is matched by a only
    assert values["b"]["PoLiS_paired"]["per_seed"] == [pytest.approx(0.5)]
    assert values["a"]["theta_err_median"]["per_seed"] == [pytest.approx(2.0)]
    assert values["b"]["theta_err_median"]["per_seed"] == [pytest.approx(5.0)]  # the pair without a θ is left out


def test_theta_errors_are_none_for_a_method_without_theta():
    pairs = pd.DataFrame({"image_id": [1], "gt_id": [10], "theta_err": [float("nan")]})
    result = tables.theta_errors(stats.RunTables(pd.DataFrame(), pairs), [1], {10})
    assert result == {"theta_err_median": None, "theta_err_p90": None}


def test_ablation_summarises_the_seeds_and_leaves_out_what_was_not_run(splits):
    install(b5="a", b5p="a", full_p0="gt")
    shutil.rmtree(metrics_root() / "full_p2", ignore_errors=True)
    payload, lines = tables.ablation("crowdai")
    assert "b1" not in payload["rows"] and "full_p2" not in payload["rows"]
    row = payload["rows"]["b5"]
    overall = read_json(paths.metrics_file("crowdai", "b5", "seed0", "test"))["overall"]
    assert row["runs"] == list(SEED_RUNS) and len(row["sources"]) == 3
    assert row["metrics"]["AP"]["mean"] == pytest.approx(overall["AP"])
    assert row["metrics"]["AP"]["sd"] == pytest.approx(0, abs=1e-12)
    single = payload["rows"]["b0"]
    assert single["runs"] == ["ref"] and single["metrics"]["AP"]["sd"] is None
    assert sum(line.startswith("| B") or line.startswith("| FULL") for line in lines) == 4  # B0, B5, B5′, FULL-P0


def test_missing_evaluations_are_named(splits):
    install(full_p0="a", full_p2="a")
    assert tables.missing_for("crowdai", "n3") == []
    assert tables.missing_for("crowdai", "ablation") == []
    file = paths.metrics_file("crowdai", "full_p2", "seed2", "test")
    file.unlink()
    assert tables.missing_for("crowdai", "n3") == [str(file)]


def script():
    spec = importlib.util.spec_from_file_location("tables_script", paths.REPO_ROOT / "scripts" / "tables.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_writes_valid_json_with_its_sources_and_stops_on_missing_runs(splits, capsys):
    install(full_p0="a", full_p2="fix_merge")
    make_dense(splits, [6])
    script().main(["--data", "crowdai", "--table", "n3", "--n-rep", "10", "--splits-dir", str(splits)])
    folder = paths.out_root() / "tables" / "crowdai"
    assert str(folder / "n3.md") in capsys.readouterr().out

    def no_nan(token):
        raise AssertionError(f"{token} in the JSON")

    record = json.loads((folder / "n3.json").read_text(encoding="utf-8"), parse_constant=no_nan)
    assert record["table"] == "n3" and record["n_rep"] == 10 and "git" in record and "created" in record
    assert [len(record["sources"][exp]) for exp in tables.N3_METHODS] == [3, 3]
    assert (folder / "n3.md").read_text(encoding="utf-8").startswith("# N3: padat, test (1 tile, 3 seed)")

    paths.metrics_file("crowdai", "full_p2", "seed1", "test").unlink()
    with pytest.raises(SystemExit):
        script().main(["--data", "crowdai", "--table", "n3", "--splits-dir", str(splits)])
    assert "do not exist" in capsys.readouterr().err
