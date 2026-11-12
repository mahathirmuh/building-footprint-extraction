"""Tables of the comparison with other methods and of the density analysis (densepoly.eval.reports), on run_eval outputs
of the synthetic scene and on numbers whose answer is known."""

import importlib.util
import json
import shutil

import pytest

from densepoly import paths
from densepoly.eval import coco_io, reports, run_eval, stats, tables
from densepoly.utils.io import read_json, write_json
from tests.fixtures.coco_toy import make_gt, make_results
from tests.fixtures.eval_scene import HOUSES, SECONDARY, STRATA, predictions, write_scene

SEED_RUNS = ("seed0", "seed1", "seed2")
PAPER = {
    "hisup": {
        "crowdai_val": {
            "method": "HiSup",
            "source": {"tables": "Table 1", "page": 9},
            "split_note": "val CrowdAI,\n  bocor",
            "values": {"AP": 79.4, "PoLiS": 0.726, "CIoU": 89.6},
        },
        "whu_test": {
            "method": "HiSup",
            "source": {"tables": "Table 3", "page": 7},
            "split_note": "WHU 512",
            "values": {"IoU": 91.2, "CIoU": 79.62, "PoLiS": 1.5},
        },
    },
    "ffl": {
        "crowdai_val": {
            "method": "FFL",
            "source": {"tables": "Table 2", "page": 6},
            "split_note": "val CrowdAI,\n  bocor",
            "values": {"AP": 61.3, "AP50": 87.5},
        }
    },
}


def metrics_root(dataset="crowdai"):
    return paths.out_root() / "metrics" / dataset


def install(dataset="crowdai", runs=SEED_RUNS, **methods):
    """Make each experiment (keyword) a copy, per run, of an evaluated method of the scene (value)."""
    for exp, source in methods.items():
        shutil.rmtree(metrics_root(dataset) / exp, ignore_errors=True)
        for run in runs:
            shutil.copytree(metrics_root(dataset) / "scene" / source, metrics_root(dataset) / exp / run)


def choose_final(exp="full_p0"):
    """val_select APs that make exp the FULL final (§8 point 6)."""
    for candidate in tables.FULL_CANDIDATES:
        for run in SEED_RUNS:
            ap = 0.7 if candidate == exp else 0.6
            write_json(paths.metrics_file("crowdai", candidate, run, "val_select"), {"overall": {"AP": ap}, "git": "abc"})


@pytest.fixture(scope="module")
def splits(tmp_path_factory):
    """Splits folder of the scene as crowdai/test, crowdai/val_clean and whu/test (also against a p2p ground truth),
    evaluated with its predictions (a), the GT itself (gt) and a with the merge of image 6 fixed (fix_merge)."""
    with pytest.MonkeyPatch.context() as mp:
        gt, folder = write_scene(tmp_path_factory.mktemp("reports"), mp)
        strata = {"main": {"strata": STRATA}, "secondary": {"strata": SECONDARY}}
        for dataset, split, variant in (("crowdai", "val_clean", None), ("whu", "test", None), ("whu", "test", "p2p")):
            write_json(paths.gt_file(dataset, split, variant), make_gt(HOUSES))
            write_json(paths.density_file(dataset, split, folder), strata)
        a = predictions()
        methods = {
            "a": a,
            "gt": coco_io.gt_as_predictions(gt),
            "fix_merge": [r for r in a if r["image_id"] != 6] + make_results({6: HOUSES[6]}),
        }
        for name, results in methods.items():
            run_eval.evaluate("crowdai", "test", "scene", name, results, splits_dir=folder)
            run_eval.evaluate("crowdai", "val_clean", "scene", name, results, splits_dir=folder)
            run_eval.evaluate("whu", "test", "scene", name, results, splits_dir=folder)
            run_eval.evaluate("whu", "test", "scene", name, results, variant="p2p", splits_dir=folder)
        yield folder


def test_strata_come_in_report_order_without_the_empty_tiles(splits):
    groups = reports.strata_image_ids("crowdai", "test", splits)
    assert list(groups) == ["jarang", "sedang", "padat", "bersentuhan", "tidak_bersentuhan"]
    assert groups["padat"] == [1] and groups["bersentuhan"] == [6]


@pytest.mark.parametrize(("delta", "supported"), [(0.02, True), (0.0, True), (-0.03, True), (-0.0301, False), (-0.1, False)])
def test_near_sota_allows_three_ap_points_behind(delta, supported):
    decision = reports.decide_sota({"delta": delta, "ci95": [delta - 0.01, delta + 0.01]})
    assert decision["supported"] is supported and decision["limit"] == pytest.approx(-0.03)


def reference_as(method):
    shutil.rmtree(metrics_root() / "hisup_release", ignore_errors=True)
    shutil.copytree(metrics_root() / "scene" / method, metrics_root() / "hisup_release" / "ref")


def test_sota_compares_the_final_with_the_released_predictions(splits):
    install(full_p0="gt")
    choose_final("full_p0")
    reference_as("a")
    payload, lines = reports.sota("crowdai", n_rep=20, splits_dir=splits)
    assert payload["final"] == "full_p0" and payload["n_images"] == 6
    assert list(payload["values"]) == ["semua", "jarang", "sedang", "padat"]
    reference = read_json(paths.metrics_file("crowdai", "hisup_release", "ref", "val_clean"))
    shown = payload["values"]["semua"]["hisup_release"]["AP"]
    assert shown["per_seed"] == [reference["overall"]["AP"]] and shown["summary"]["sd"] is None
    assert payload["values"]["padat"]["hisup_release"]["IoU"]["per_seed"] == [reference["per_stratum"]["padat"]["IoU"]]
    assert payload["values"]["semua"]["full_p0"]["AP"]["summary"]["mean"] == pytest.approx(1.0)
    ours = tables.load_runs("crowdai", ("full_p0",), "val_clean")["full_p0"]
    ref = stats.load_run("crowdai", "hisup_release", "ref", "val_clean")
    expected = stats.compare([ref] * 3, ours, "AP", [1, 2, 3, 4, 5, 6], 20, tables.BOOTSTRAP_SEED)
    assert payload["deltas"]["semua"]["AP"] == expected and expected["delta"] > 0
    assert payload["deltas"]["padat"]["IoU"]["delta"] > 0  # a misses a house of image 1 and shifts the others
    assert payload["decision"]["supported"] is True
    assert [len(payload["sources"][exp]) for exp in ("hisup_release", "full_p0")] == [1, 3]
    assert any('**"Mendekati SOTA" didukung**' in line for line in lines)
    assert any(line.startswith("| HiSup | ya |") for line in lines)


def test_sota_is_not_near_when_the_final_is_far_behind(splits):
    install(full_p2="a")
    choose_final("full_p2")
    reference_as("gt")
    payload, lines = reports.sota("crowdai", n_rep=20, splits_dir=splits)
    assert payload["final"] == "full_p2"
    assert payload["decision"]["delta"] < -0.03 and payload["decision"]["supported"] is False
    assert any('**"Mendekati SOTA" tidak didukung**' in line for line in lines)


def test_sota_names_what_is_missing(splits):
    install(full_p0="gt")
    choose_final("full_p0")
    reference_as("a")
    assert tables.missing_for("crowdai", "sota") == []
    file = paths.metrics_file("crowdai", "full_p0", "seed1", "val_clean")
    file.unlink()
    assert tables.missing_for("crowdai", "sota") == [str(file)]
    assert tables.missing_for("whu", "sota") != []


def test_paper_table_prints_the_numbers_as_reported(monkeypatch):
    monkeypatch.setattr(reports, "load_paper_numbers", lambda: PAPER)
    payload, lines = reports.paper("crowdai")
    assert payload["columns"] == ["AP", "AP50", "CIoU", "PoLiS"]
    assert "| HiSup | 79,4 | — | 89,6 | 0,726 | Table 1; hlm. 9 |" in lines
    assert "| FFL | 61,3 | 87,5 | — | — | Table 2; hlm. 6 |" in lines
    assert lines.count("- val CrowdAI, bocor") == 1  # the same note once
    assert tables.trace(payload) == ["configs/paper_numbers.yaml"]
    assert list(reports.paper("whu")[0]["rows"]) == ["hisup"]
    monkeypatch.setattr(reports, "load_paper_numbers", lambda: {"ffl": PAPER["ffl"]})
    assert tables.missing_for("whu", "paper") != [] and tables.missing_for("crowdai", "paper") == []


def test_strata_table_reads_the_values_per_stratum(splits):
    for exp in tables.LABELS:
        shutil.rmtree(metrics_root() / exp, ignore_errors=True)
    assert tables.missing_for("crowdai", "strata") != []
    install(b4="a", full_p0="fix_merge")
    payload, lines = reports.strata("crowdai", splits_dir=splits)
    assert list(payload["rows"]) == ["b4", "full_p0"]
    record = read_json(paths.metrics_file("crowdai", "b4", "seed0", "test"))
    touching = payload["rows"]["b4"]["strata"]["bersentuhan"]["merge_rate"]
    assert touching["per_seed"] == [record["per_stratum"]["bersentuhan"]["merge_rate"]] * 3
    assert touching["summary"]["mean"] == pytest.approx(1.0)  # both houses of image 6 under one polygon
    assert payload["rows"]["full_p0"]["strata"]["bersentuhan"]["merge_rate"]["summary"]["mean"] == pytest.approx(0)
    assert payload["sizes"]["padat"] == {"n_images": 1, "n_gt": 6}
    assert "## merge_rate" in lines and any("bersentuhan (1 tile)" in line for line in lines)


@pytest.mark.parametrize(("ci", "passed"), [([-0.3, -0.1], True), ([-0.3, 0.0], False), ([0.1, 0.3], False), (None, False)])
def test_larger_in_dense_needs_the_ci_below_zero(ci, passed):
    assert reports.larger_in_dense({"difference": -0.2, "ci95": ci})["passed"] is passed


def test_decide_n2_needs_the_claim_itself_and_every_metric():
    yes, no = {"passed": True}, {"passed": False}
    assert reports.decide_n2(True, {"a": yes, "b": yes})["supported"] is True
    assert reports.decide_n2(True, {"a": yes, "b": no})["supported"] is False
    assert reports.decide_n2(False, {"a": yes, "b": yes})["supported"] is False
    assert reports.decide_n2(None, {"a": yes})["supported"] is False


def test_n2_follows_each_claim_over_the_strata(splits):
    install(b5="a", b5p="a", full_p0="gt", full_p2="gt")
    payload, lines = reports.n2("crowdai", n_rep=20, splits_dir=splits)
    assert list(payload["comparisons"]) == ["b5->full_p0", "full_p0->full_p2"]
    first = payload["comparisons"]["b5->full_p0"]
    runs = tables.load_runs("crowdai", ("b5", "full_p0"), "test")
    assert first["per_stratum"]["padat"]["PoLiS_paired"] == stats.compare(
        runs["b5"], runs["full_p0"], "PoLiS_paired", [1], 20, tables.BOOTSTRAP_SEED
    )
    assert first["dense_minus_sparse"]["MTA_inst"] == stats.compare_strata(
        runs["b5"], runs["full_p0"], "MTA_inst", [1], [3, 4], 20, tables.BOOTSTRAP_SEED
    )
    assert first["per_stratum"]["padat"]["PoLiS_paired"]["delta"] < 0
    assert first["n_images"] == {"jarang": 2, "sedang": 2, "padat": 1, "bersentuhan": 1, "tidak_bersentuhan": 4}
    assert first["decision"]["base_supported"] == tables.n1("crowdai", 20, splits)[0]["decision"]["supported"]
    second = payload["comparisons"]["full_p0->full_p2"]  # identical predictions: nothing to claim
    assert second["per_stratum"]["padat"]["merge_rate"]["delta"] == 0
    assert second["decision"]["base_supported"] is False and second["decision"]["supported"] is False
    assert any('"Efek N3 terbesar di area padat" tidak didukung' in line for line in lines)
    assert any(line.startswith("- Δpadat − Δjarang, `PoLiS_paired`") for line in lines)


def test_n2_on_whu_is_a_description_of_one_seed(splits):
    install("whu", runs=("seed0",), b5p="a", full_p0="gt")
    assert tables.missing_for("whu", "n2") == []
    payload, lines = reports.n2("whu", n_rep=20, splits_dir=splits)
    step = payload["comparisons"]["b5p->full_p0"]
    assert step["seeds"] == [0] and len(step["per_stratum"]["padat"]["PoLiS_paired"]["delta_per_seed"]) == 1
    assert step["decision"]["base_supported"] is None and step["decision"]["supported"] is False
    assert any("Deskriptif saja" in line for line in lines)
    paths.metrics_file("whu", "b5p", "seed0", "test").unlink()
    assert tables.missing_for("whu", "n2") != []


def test_p2p_table_puts_our_whu_numbers_next_to_the_paper(splits, monkeypatch):
    monkeypatch.setattr(reports, "load_paper_numbers", lambda: PAPER)
    install("whu", runs=("seed0",), full_p0="a")
    assert tables.missing_for("whu", "p2p") == [] and tables.missing_for("crowdai", "p2p") != []
    payload, lines = reports.p2p("whu")
    assert list(payload["ours"]) == ["full_p0/p2p", "full_p0/utama"]
    record = read_json(paths.metrics_file("whu", "full_p0", "seed0", "test", "p2p"))
    assert payload["ours"]["full_p0/p2p"]["metrics"]["CIoU_p2p"]["per_seed"] == [record["overall"]["CIoU_p2p"]]
    assert payload["ours"]["full_p0/p2p"]["sources"][0]["file"].endswith("whu/full_p0/seed0/test_p2p.json")
    ours = tables.cell("CIoU", record["overall"]["CIoU_p2p"])
    assert any(line.startswith("| FULL-P0 (kita, 1 run) | p2p |") and ours in line for line in lines)
    assert "| HiSup | p2p (paper) | 91,2 | 79,62 | — | 1,5 | — | dilaporkan di paper: Table 3; hlm. 7 |" in lines
    assert any("hanya kira-kira" in line for line in lines)


def test_efficiency_table_shows_the_benchmark(splits):
    assert tables.missing_for("crowdai", "efficiency") != []
    record = {
        "parameters": 20_812_345,
        "input": [1, 3, 320, 320],
        "fps": 100.0,
        "network_ms": 10.0,
        "postproc": {"total_ms": 15.0, "instances_ms": 4.0, "polygons_ms": 11.0},
        "gpu": "GPU X",
        "cpu": "CPU Y",
    }
    write_json(paths.efficiency_file("crowdai", "full_p0", "seed0"), record)
    assert tables.missing_for("crowdai", "efficiency") == []
    payload, lines = reports.efficiency("crowdai")
    assert "| FULL-P0 | 20,81 | 320×320 | 100,0 | 10,00 | – | 15,00 | 40,0 |" in lines
    assert "- Perangkat: GPU GPU X; CPU CPU Y" in lines
    assert tables.trace(payload) == ["metrics/crowdai/full_p0/seed0/efficiency.json"]


def test_efficiency_table_adds_the_detector_time_of_seeded_experiments(splits):
    record = {
        "parameters": 20_812_345,
        "input": [1, 3, 320, 320],
        "fps": 100.0,
        "network_ms": 10.0,
        "postproc": {"total_ms": 15.0, "instances_ms": 4.0, "polygons_ms": 11.0},
        "detector": {"seconds_per_image": 0.025},
        "gpu": "GPU X",
        "cpu": "CPU Y",
    }
    write_json(paths.efficiency_file("crowdai", "full_y", "seed0"), record)
    _, lines = reports.efficiency("crowdai")
    assert "| FULL+Y | 20,81 | 320×320 | 100,0 | 10,00 | 25,00 | 15,00 | 20,0 |" in lines


def script():
    spec = importlib.util.spec_from_file_location("tables_script", paths.REPO_ROOT / "scripts" / "tables.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_writes_the_new_tables_and_traces_their_sources(splits, capsys):
    install(b4="a", full_p0="fix_merge")
    script().main(["--data", "crowdai", "--table", "strata", "--splits-dir", str(splits)])
    folder = paths.out_root() / "tables" / "crowdai"
    assert str(folder / "strata.md") in capsys.readouterr().out
    record = json.loads((folder / "strata.json").read_text(encoding="utf-8"))
    assert record["table"] == "strata" and "git" in record
    script().main(["--data", "crowdai", "--table", "strata", "--trace"])
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "strata:" and "  metrics/crowdai/b4/seed0/test.json" in out
    assert "  metrics/crowdai/full_p0/seed2/test.json" in out


def test_result_figures_are_drawn_from_the_written_tables(splits):
    install(b3="a", b4="a", b5="a", b5p="a", full_p0="gt", full_p2="gt")
    figures = importlib.util.spec_from_file_location("figures_script", paths.REPO_ROOT / "scripts" / "figures.py")
    module = importlib.util.module_from_spec(figures)
    figures.loader.exec_module(module)
    with pytest.raises(FileNotFoundError, match="scripts/tables.py"):
        module.read_table("crowdai", "no_such_table")
    tables.write_table("crowdai", "strata", splits_dir=splits)
    tables.write_table("crowdai", "n2", n_rep=20, splits_dir=splits)
    module.main(["--data", "crowdai", "--results", "--lang", "id"])
    folder = paths.out_root() / "figures" / "results"
    for name in ("crowdai_n2_id.png", "crowdai_n2_id.pdf", "crowdai_strata_id.png", "crowdai_strata_id.pdf"):
        assert (folder / name).stat().st_size > 1000
    assert module.number(-0.0371, module.WORDS["id"], 3, signed=True) == "−0,037"
    assert module.number(1.5, module.WORDS["en"]) == "1.50"
    apart = module.spread([1.0, 5.0, 1.2], 1.0)
    assert apart[2] - apart[0] == pytest.approx(1.0) and apart[1] - apart[2] >= 1.0  # order kept, neighbours a gap apart
    assert sum(apart) == pytest.approx(7.2)  # the mean stays
    assert module.spread([3.0, 1.0], 0.5) == [3.0, 1.0]  # far enough already
