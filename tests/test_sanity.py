"""Sanity checks of the evaluation pipeline (densepoly.eval.sanity; docs/EVALUATION.md §6.1, §6.2, §6.4, §6.6)."""

import importlib.util

import numpy as np
import pytest

from densepoly import paths
from densepoly.eval import coco_io, instance, sanity
from densepoly.utils.io import write_json
from tests.fixtures.coco_toy import closed_ring, make_gt, square
from tests.fixtures.eval_scene import write_scene


def overlapping_gt():
    """House 2 lies inside house 1; houses 4 and 5 are the same polygon twice; house 3 stands alone."""
    return make_gt({1: [square(10, 10, 20), square(12, 12, 10), square(40, 40, 10)], 2: [square(40, 5, 10)] * 2})


def test_overlap_sets_predict_the_merges_and_splits_of_gt_as_predictions():
    gt = overlapping_gt()
    covered, covering = sanity.gt_overlap_sets(gt)
    assert (covered, covering) == ({2, 4, 5}, {1, 4, 5})
    gt_rows, _ = instance.instance_metrics(gt, coco_io.gt_as_predictions(gt))
    assert {r["gt_id"] for r in gt_rows if r["merged"]} == covered | covering
    assert {r["gt_id"] for r in gt_rows if r["split"]} == covering


def test_perturbed_predictions_shift_drop_and_skip_empty_masks():
    houses = {i: [square(5 + (j % 4) * 14, 5 + (j // 4) * 14, 10) for j in range(16)] for i in range(1, 16)}
    gt = make_gt(houses)
    gt["annotations"][0]["segmentation"] = [closed_ring([(1, 1), (3, 3), (5, 5)])]  # an empty mask
    results = sanity.perturbed_predictions(gt, seed=0)
    assert results == sanity.perturbed_predictions(gt, seed=0)
    assert 0.8 * 240 < len(results) < 0.97 * 240  # ±10% of 240 houses dropped, the empty one skipped
    by_image = {}
    for ann in gt["annotations"][1:]:
        by_image.setdefault(ann["image_id"], []).append(np.reshape(ann["segmentation"][0], (-1, 2)))
    for r in results:
        ring = np.reshape(r["segmentation"][0], (-1, 2))
        shifts = [ring - gt_ring for gt_ring in by_image[r["image_id"]] if gt_ring.shape == ring.shape]
        assert any((np.ptp(s, axis=0) == 0).all() and (np.abs(s[0]) <= 2).all() for s in shifts)


def test_gt_as_prediction_passes_with_an_empty_mask(tmp_path, monkeypatch):
    """§6.1 end to end on the toy scene plus a zero-area GT ring: (ii) leaves exactly it unmatched, (i) is ideal."""
    gt, _ = write_scene(tmp_path, monkeypatch)
    gt["annotations"].append(
        {**gt["annotations"][0], "id": 99, "segmentation": [closed_ring([(1, 1), (3, 3), (5, 5)])]}
    )
    write_json(paths.gt_file("crowdai", "test"), gt)
    part = sanity.gt_as_prediction("crowdai", "test")
    assert part["empty_masks"] == 1
    assert all(c["passed"] for c in part["checks"]), [c for c in part["checks"] if not c["passed"]]
    assert part["as_released"]["AP"] < 1 and part["without_empty"]["AP"] == pytest.approx(1.0)


def test_determinism_and_bootstrap_on_the_toy_scene(tmp_path, monkeypatch):
    gt, _ = write_scene(tmp_path, monkeypatch)
    write_json(paths.gt_file("crowdai", "val_clean"), gt)
    assert sanity.determinism()["checks"][0]["passed"]
    part = sanity.bootstrap_ap(n_rep=60, n_naive=20)
    identity, contains, _ = part["checks"]
    assert identity["passed"] and contains["passed"]
    assert 0 < part["naive_unique_fraction_mean"] < 1


def test_report_renders_every_check():
    check = sanity.check("6.1", "crowdai/test", "(i) AP", 1.0, "1 ± 0,001", True)
    part = {"case": "crowdai/test", "images": 3, "annotations": 5, "empty_masks": 0,
            "pred_check_as_released": {"dropped": 0, "too_few_points": 0}, "checks": [check]}  # fmt: skip
    bootstrap = {"case": "crowdai/val_clean", "point_AP": 0.5, "n_rep": 10, "ci95": [0.4, 0.6], "sd": 0.05,
                 "n_naive": 5, "naive_unique_fraction_mean": 0.63, "naive_ci95": [0.45, 0.55],
                 "proper_ci95_first_n_naive": [0.4, 0.6], "width_ratio": 0.5,
                 "checks": [sanity.check("6.6", "crowdai/val_clean", "x", 1, "y", False)]}  # fmt: skip
    report = {"created": "now", "git": "abc", "environment": sanity.environment(), "gt_as_prediction": [part],
              "edge_case_tests": [{"test": "tests/test_x.py::test_y", "passed": True}],
              "determinism": {"checks": [sanity.check("6.2", "crowdai/val_clean", "same", True, "identik", True)]},
              "bootstrap_ap": bootstrap}  # fmt: skip
    text = sanity.render_report(report)
    assert (
        "**Hasil: 2 dari 3 uji lolos.**" in text
        and "**gagal**" in text
        and "| (i) AP | 1 | 1 ± 0,001 | lolos |" in text
    )
    assert "Belum ada: evaluasi `hisup_release`" in text  # §6.4 before HiSup's release is evaluated


def bare_report(hisup: dict | None = None) -> dict:
    """A report whose only part with checks is §6.4."""
    return {"created": "now", "git": "abc", "environment": sanity.environment(), "gt_as_prediction": [],
            "edge_case_tests": [], "determinism": {"checks": []}, "hisup_reproduction": hisup,
            "bootstrap_ap": {"case": "c", "point_AP": 0.5, "n_rep": 1, "ci95": [0, 1], "sd": 0, "n_naive": 1,
                             "naive_unique_fraction_mean": 0.6, "naive_ci95": [0, 1],
                             "proper_ci95_first_n_naive": [0, 1], "width_ratio": 1.0, "checks": []}}  # fmt: skip


def fake_hisup_record(ap: float) -> dict:
    overall = {"AP": ap, "AP50": 0.926, "AP75": 0.85, "AP_boundary": 0.66, "PoLiS": 0.73, "CIoU_p2p": 0.89,
               "IoU_p2p": 0.94, "CIoU": 0.88, "IoU": 0.93, "MTA": 30.0}  # fmt: skip
    return {"overall": overall, "coco_stats": {"segm": {"AR100": 0.81}}, "created": "now", "git": "abc",
            "pred_check": {"results": 10, "dropped": 0, "too_few_points": 1}, "seconds": {}}  # fmt: skip


@pytest.mark.parametrize(("ap", "passed"), [(0.792, True), (0.797, True), (0.790, False), (0.7975, False)])
def test_hisup_reproduction_checks_ap_within_0_3_points(tmp_path, monkeypatch, ap, passed):
    monkeypatch.setenv("DENSEPOLY_OUT", str(tmp_path / "outputs"))
    assert sanity.hisup_reproduction() is None
    write_json(paths.metrics_file("crowdai", "hisup_release", "ref", "val_full"), fake_hisup_record(ap))
    part = sanity.hisup_reproduction()
    rows = {r["metric"]: r for r in part["rows"]}
    assert rows["AP"]["paper"] == 79.4 and rows["AP"]["ours"] == pytest.approx(100 * ap)
    assert rows["AR"]["ours"] == pytest.approx(81.0) and rows["PoLiS"]["ours"] == pytest.approx(0.73)
    assert part["checks"][0]["passed"] is passed
    text = sanity.render_report(bare_report(part))
    assert f"| AP | 79.4 | {100 * ap:.2f} |" in text
    assert "| PoLiS | 0.726 | 0.7300 | +0.0040 |" in text  # one decimal more than the paper prints


def test_render_only_renders_the_saved_report_again(tmp_path, monkeypatch):
    monkeypatch.setenv("DENSEPOLY_OUT", str(tmp_path / "outputs"))
    spec = importlib.util.spec_from_file_location("sanity_checks", paths.REPO_ROOT / "scripts" / "sanity_checks.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    report = bare_report()
    write_json(paths.out_root() / "metrics" / "sanity_report.json", report)
    md = tmp_path / "sanity_report.md"
    script.main(["--render-only", "--report-md", str(md)])
    assert md.read_text(encoding="utf-8") == sanity.render_report(report)
