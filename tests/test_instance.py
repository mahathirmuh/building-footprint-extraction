"""Instance metrics, vertex F1 and θ error (densepoly.eval.instance, vertex, orientation; docs/EVALUATION.md §2.1, §2.4,
and the synthetic checks of §6.1)."""

import math

import numpy as np
import pandas as pd
import pytest

from densepoly.eval import coco_io, instance
from densepoly.eval.orientation import polygon_theta_deg, theta_error_deg
from densepoly.eval.vertex import ring_vertices, vertex_matches
from tests.fixtures.coco_toy import RECT, closed_ring, make_gt, make_results, rotated, square


def evaluate(gt, results, workers=1):
    gt_rows, pair_rows = instance.instance_metrics(gt, results, workers)
    return gt_rows, pair_rows, instance.summarize_instance(gt_rows, pair_rows)


def test_matching_is_greedy_by_score_and_one_to_one():
    ious = np.array([[0.9, 0.6], [0.7, 0.2]])
    assert instance.match(ious, [0.5, 0.9]) == [(1, 0, 0.7), (0, 1, 0.6)]  # the higher score picks first
    assert instance.match(np.array([[0.49, 0.3]]), [1.0]) == []
    assert instance.match(np.array([[0.8, 0.8]]), [1.0]) == [(0, 0, 0.8)]  # exact tie: first GT
    assert instance.match(np.array([[0.9], [0.95]]), [0.9, 0.8]) == [(0, 0, 0.9)]  # one GT, one pair
    assert instance.match(np.zeros((0, 3)), []) == [] and instance.match(np.zeros((2, 0)), [1, 1]) == []


def test_gt_as_prediction_gives_the_ideal_values():
    """§6.1 on a small scene, with θ_pred equal to θ_GT."""
    houses = {1: [square(5, 5, 10), rotated(RECT, 30)], 2: [square(20, 20, 16)]}
    gt = make_gt(houses)
    results = coco_io.gt_as_predictions(gt)
    for result, ann in zip(results, gt["annotations"], strict=True):
        result["theta_deg"] = polygon_theta_deg(ann["segmentation"][0])
    _, pairs, summary = evaluate(gt, results)
    assert (summary["recall_inst50"], summary["merge_rate"], summary["split_rate"]) == (1.0, 0.0, 0.0)
    assert summary["PoLiS_paired"] == 0.0 and summary["vertex_F1"] == 1.0
    assert 0 <= summary["MTA_inst"] < 0.01
    assert summary["theta_err_median"] == pytest.approx(0, abs=1e-9)
    assert sorted(p["gt_id"] for p in pairs) == [1, 2, 3]


@pytest.mark.parametrize("d", [1.0, 2.0])
def test_polis_paired_of_a_shifted_square_is_0_4_d(d):
    """§6.1: PoLiS = PoLiS_paired = 0.4·d for an axis-aligned square shifted by d px along x."""
    gt = make_gt({1: [square(10, 10, 20)]})
    _, pairs, summary = evaluate(gt, make_results({1: [[(x + d, y) for x, y in square(10, 10, 20)]]}))
    assert summary["PoLiS_paired"] == pytest.approx(0.4 * d)
    assert summary["vertex_F1"] == 1.0  # every vertex within 2 px


def test_vertex_f1_needs_vertices_within_2_px():
    gt = make_gt({1: [square(10, 10, 20)]})
    _, _, summary = evaluate(gt, make_results({1: [square(13, 10, 20)]}))
    assert summary["n_pairs"] == 1 and summary["vertex_F1"] == 0.0


@pytest.mark.parametrize("alpha", [5, 10, 20])
def test_mta_inst_of_a_rotated_polygon(alpha):
    """§6.1: rotating by alpha about the centre gives MTA_inst ≥ alpha − 0.5°."""
    gt = make_gt({1: [RECT]})
    _, _, summary = evaluate(gt, make_results({1: [rotated(RECT, alpha)]}))
    assert summary["n_pairs"] == 1 and summary["MTA_inst"] >= alpha - 0.5


def test_one_prediction_over_two_houses_merges_them():
    gt = make_gt({1: [square(5, 5, 10), square(16, 5, 10), square(40, 40, 10)]})
    gt_rows, _, summary = evaluate(gt, make_results({1: [[(5, 5), (26, 5), (26, 15), (5, 15)]]}))
    assert [r["merged"] for r in gt_rows] == [True, True, False]
    assert summary["merge_rate"] == pytest.approx(2 / 3) and summary["recall_inst50"] == 0.0


def test_two_predictions_inside_one_house_split_it():
    gt = make_gt({1: [square(10, 10, 20)]})
    halves = [[(10, 10), (22, 10), (22, 30), (10, 30)], [(22, 10), (30, 10), (30, 30), (22, 30)]]  # IoU 0.6, 0.4
    gt_rows, pairs, summary = evaluate(gt, make_results({1: halves}))
    assert gt_rows[0]["split"] and summary["split_rate"] == 1.0
    assert summary["recall_inst50"] == 1.0 and pairs[0]["pred_index"] == 0 and pairs[0]["iou"] == pytest.approx(0.6)


def test_a_gt_with_an_empty_mask_only_counts_against_recall():
    gt = make_gt({1: [square(10, 10, 20)]})
    gt["annotations"].append({**gt["annotations"][0], "id": 2, "segmentation": [closed_ring([(1, 1), (3, 3), (5, 5)])]})
    gt_rows, _, summary = evaluate(gt, make_results({1: [square(10, 10, 20), [(1, 1), (6, 1), (6, 6)]]}))
    empty = gt_rows[1]
    assert (empty["gt_px"], empty["matched"], empty["merged"], empty["split"]) == (0, False, False, False)
    assert summary["recall_inst50"] == 0.5


def test_theta_error_uses_theta_deg_and_wraps_at_90():
    gt = make_gt({1: [square(10, 10, 20)]})  # θ_GT = 0
    _, pairs, summary = evaluate(gt, make_results({1: [square(10, 10, 20)]}, theta_deg=89.0))
    assert pairs[0]["theta_err"] == pytest.approx(1.0) and summary["theta_err_median"] == pytest.approx(1.0)
    _, pairs, summary = evaluate(gt, make_results({1: [square(10, 10, 20)]}))  # no θ stored
    assert math.isnan(pairs[0]["theta_err"]) and summary["theta_err_median"] is None


def test_workers_give_the_same_rows():
    houses = {i: [square(5 + i, 5, 10), rotated(RECT, 3 * i)] for i in range(1, 6)}
    gt = make_gt(houses)
    results = make_results({i: [square(6 + i, 5, 10), rotated(RECT, 3 * i + 4)] for i in range(1, 6)})
    parallel = instance.instance_metrics(gt, results, workers=2)
    serial = instance.instance_metrics(gt, results, workers=1)
    for a, b in zip(parallel, serial, strict=True):  # NaN θ errors: compare as frames, where NaN equals NaN
        pd.testing.assert_frame_equal(pd.DataFrame(a), pd.DataFrame(b))


def test_no_prediction_and_no_gt():
    _, _, summary = evaluate(make_gt({1: [square(10, 10, 20)]}), [])
    assert summary["recall_inst50"] == 0.0 and summary["n_pairs"] == 0 and summary["PoLiS_paired"] is None
    _, _, summary = evaluate(make_gt({1: []}), make_results({1: [square(10, 10, 20)]}))
    assert summary["recall_inst50"] is None and summary["n_gt"] == 0


def test_vertex_matches_maximise_the_number_of_pairs():
    gt, pred = np.array([[0.0, 0.0], [3.0, 0.0]]), np.array([[1.5, 0.0], [4.5, 0.0]])
    assert vertex_matches(gt, pred) == 2  # nearest-first would pair (3, 0) with (1.5, 0) and leave one out
    assert vertex_matches(gt, pred + [0.0, 2.5]) == 0
    assert vertex_matches(gt, np.zeros((0, 2))) == 0
    assert ring_vertices(closed_ring(square(0, 0, 4))).shape == (4, 2)


def test_polygon_theta_and_its_error():
    assert polygon_theta_deg(closed_ring(RECT)) == pytest.approx(0)
    assert polygon_theta_deg(closed_ring([(y, x) for x, y in RECT])) == pytest.approx(0)  # long side vertical
    assert polygon_theta_deg(closed_ring(rotated(RECT, 30))) == pytest.approx(30, abs=1e-3)
    assert polygon_theta_deg(closed_ring(rotated(RECT, -30))) == pytest.approx(60, abs=1e-3)
    assert [theta_error_deg(a, b) for a, b in [(89, 1), (44, 0), (46, 0), (135, 0), (-10, 80)]] == [2, 44, 44, 45, 0]
