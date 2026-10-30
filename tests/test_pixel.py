"""Pixel metrics of union masks (densepoly.eval.pixel; docs/EVALUATION.md §2.1, §2.2, §6.2)."""

import json

import numpy as np
import pytest
from pix2poly_metrics import cIoU

from densepoly.eval import pixel
from tests.fixtures.coco_toy import SIZE, make_gt, make_results, square


def rows_by_image(gt, results):
    return {row["image_id"]: row for row in pixel.pixel_metrics(gt, results)}


def test_a_perfect_prediction_scores_one_everywhere():
    gt = make_gt({1: [square(10, 10, 20)]})
    row = rows_by_image(gt, make_results({1: [square(10, 10, 20)]}))[1]
    assert (row["gt_px"], row["pred_px"], row["inter_px"]) == (400, 400, 400)
    assert row["iou"] == pytest.approx(1.0) and row["ciou"] == pytest.approx(1.0) and row["biou"] == 1.0
    assert (row["n_vertices"], row["n_vertices_gt"]) == (5, 5)  # closing vertex counted, as in cIoU.py


def test_edge_cases_of_docs_evaluation_6_2():
    gt = make_gt({1: [square(10, 10, 20)], 2: [], 3: []})
    rows = rows_by_image(gt, make_results({3: [square(5, 5, 10)]}))
    no_prediction, both_empty, no_gt = rows[1], rows[2], rows[3]
    assert (no_prediction["iou"], no_prediction["ciou"], no_prediction["biou"]) == (0.0, 0.0, 0.0)
    assert (both_empty["iou"], both_empty["ciou"], both_empty["biou"]) == (1.0, 1.0, 1.0)
    assert (no_gt["iou"], no_gt["ciou"], no_gt["biou"]) == (0.0, 0.0, 0.0)


def test_pooled_and_per_image_iou_differ_as_defined():
    gt = make_gt({1: [square(0, 0, 40)], 2: [square(10, 10, 4)]})
    results = make_results({1: [square(0, 0, 40)], 2: [square(12, 10, 4)]})  # image 2: half of the house
    rows = pixel.pixel_metrics(gt, results)
    summary = pixel.summarize_pixel(rows)
    assert summary["IoU"] == pytest.approx((1600 + 8) / (1600 + 24))
    assert summary["IoU_per_image"] == pytest.approx((1 + 8 / 24) / 2)
    assert summary["F1"] == pytest.approx(2 * (1600 + 8) / (1616 + 1616))
    assert summary["CIoU"] == pytest.approx(np.mean([r["ciou"] for r in rows]))
    assert summary["N_ratio"] == pytest.approx(1.0) and summary["n_images_with_pred"] == 2


def test_ciou_equals_the_vendored_script_on_images_with_predictions(tmp_path):
    gt = make_gt({1: [square(10, 10, 20)], 2: [square(5, 5, 12), square(35, 35, 15)], 3: [square(20, 20, 16)], 4: []})
    results = make_results({1: [square(12, 10, 20)], 2: [square(5, 5, 12), square(45, 5, 8)], 4: [square(9, 9, 9)]})
    (tmp_path / "gt.json").write_text(json.dumps(gt))
    (tmp_path / "dt.json").write_text(json.dumps(results))
    vendored = cIoU.compute_IoU_cIoU(str(tmp_path / "dt.json"), str(tmp_path / "gt.json"), progress=False)
    rows = rows_by_image(gt, results)
    for image_id, values in vendored["per_image"].items():
        assert rows[image_id]["ciou"] == pytest.approx(values["ciou"], abs=1e-12)
        assert rows[image_id]["iou"] == pytest.approx(values["iou"], abs=1e-12)
    with_predictions = [rows[i]["ciou"] for i in vendored["per_image"]]
    assert vendored["ciou"] == pytest.approx(np.mean(with_predictions), abs=1e-12)  # CIoU_p2p
    # The main CIoU also counts image 3, which has GT but no prediction, as 0.
    assert pixel.summarize_pixel(list(rows.values()))["CIoU"] == pytest.approx(sum(with_predictions) / 4)


def test_boundary_iou_is_stricter_than_mask_iou():
    gt = np.zeros((SIZE, SIZE), bool)
    gt[10:40, 10:40] = True
    shifted = np.roll(gt, 2, axis=1)
    iou = (gt & shifted).sum() / (gt | shifted).sum()
    assert 0 < pixel.boundary_iou(gt, shifted) < iou
    assert pixel.boundary_iou(gt, gt) == 1.0
    assert pixel.boundary_iou(gt, np.roll(gt, 32, axis=0)) == 0.0
    assert pixel.boundary_iou(np.zeros_like(gt), np.zeros_like(gt)) == 1.0
    assert pixel.boundary_iou(gt, np.zeros_like(gt)) == 0.0


def test_summary_of_no_image():
    summary = pixel.summarize_pixel([])
    assert summary["n_images"] == 0 and all(summary[k] is None for k in ("IoU", "F1", "CIoU", "BoundaryIoU"))
