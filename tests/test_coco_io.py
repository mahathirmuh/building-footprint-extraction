"""Ground truth per split and prediction checks (densepoly.eval.coco_io; docs/EVALUATION.md §1, §2.2)."""

import contextlib
import importlib.util
import io
import json

import numpy as np
import pytest
from pycocotools.cocoeval import COCOeval

from densepoly import paths
from densepoly.eval import coco_io
from densepoly.utils.io import read_json, write_lines
from tests.fixtures.fake_crowdai import make_fake_crowdai

SIZE = 64
CATEGORY = 100


def square(x, y, side):
    return [(x, y), (x + side, y), (x + side, y + side), (x, y + side)]


def closed_ring(points):
    return [float(v) for point in [*points, points[0]] for v in point]


def ann(ann_id, image_id, rings, area=1.0):
    return {"id": ann_id, "image_id": image_id, "category_id": CATEGORY, "segmentation": rings, "area": area,
            "bbox": [], "iscrowd": 0}  # fmt: skip


def source_coco():
    """Ids from 0 (as CrowdAI val and the WHU conversion), a two-ring building, a zero-area ring, a WHU negative
    sample and an image without annotations."""
    images = [{"id": i, "file_name": f"img{i}.png", "height": SIZE, "width": SIZE} for i in (1, 2, 3, 4)]
    annotations = [
        ann(0, 1, [closed_ring(square(5, 5, 10))]),
        ann(1, 1, [closed_ring(square(30, 30, 12)), closed_ring(square(33, 33, 4))]),
        ann(2, 2, [closed_ring([(10, 10), (15, 15), (20, 20)])], area=0.0),  # zero area: covers no pixel
        ann(3, 2, [closed_ring(square(40, 8, 15))]),
        ann(4, 3, []),  # negative sample
    ]
    return {"images": images, "annotations": annotations, "categories": [{"id": CATEGORY, "name": "building"}]}


def evaluate_ap(gt, results):
    """COCO mask AP of results against gt (dicts), pycocotools as is."""
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = coco_io.coco_from_dict(gt)
        evaluator = COCOeval(coco_gt, coco_io.load_results(coco_gt, results), "segm")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return evaluator.stats[0]


def test_subset_keeps_every_image_and_drops_negative_samples():
    subset, counts = coco_io.subset_gt(source_coco(), ["img3.png", "img1.png", "img4.png"], offset=1)
    assert [img["id"] for img in subset["images"]] == [3, 1, 4]  # split-file order, empty images included
    assert [a["id"] for a in subset["annotations"]] == [1, 2]  # ids + 1, the negative sample of image 3 dropped
    assert counts == {
        "images": 3,
        "images_without_annotations": 2,
        "annotations": 2,
        "dropped_negative_samples": 1,
        "ann_id_offset": 1,
        "multi_ring_annotations": 1,
        "zero_area_rings": 0,
        "empty_masks": 0,
        "area_field_zero": 0,
        "closing_vertex_repeated": 3,
        "closing_vertex_not_repeated": 0,
        "ring_convention": "closed",
    }


def test_subset_keeps_zero_area_rings_and_counts_them():
    subset, counts = coco_io.subset_gt(source_coco(), None, offset=1)
    assert len(subset["images"]) == 4 and len(subset["annotations"]) == 4
    assert (counts["zero_area_rings"], counts["empty_masks"], counts["area_field_zero"]) == (1, 1, 1)
    assert coco_io.empty_mask_ids(subset) == [3]  # source id 2, shifted
    kept = coco_io.without_annotations(subset, [3])
    assert [a["id"] for a in kept["annotations"]] == [1, 2, 4] and kept["images"] == subset["images"]


def test_subset_rejects_images_missing_from_the_ground_truth():
    with pytest.raises(ValueError, match="not in the ground truth"):
        coco_io.subset_gt(source_coco(), ["img1.png", "nope.png"], offset=0)


def test_id_offset():
    assert coco_io.id_offset(source_coco()) == 1
    assert coco_io.id_offset({"annotations": [ann(1, 1, []), ann(5, 1, [])]}) == 0
    assert coco_io.id_offset({"annotations": []}) == 0
    with pytest.raises(ValueError, match="unique"):
        coco_io.id_offset({"annotations": [ann(1, 1, []), ann(1, 2, [])]})


def test_a_gt_with_id_0_breaks_cocoeval_and_the_offset_repairs_it():
    """COCOeval reads a match with GT id 0 as "not matched" (docs/EVALUATION.md §2.2)."""
    gt = {
        "images": [{"id": 1, "file_name": "a.png", "height": SIZE, "width": SIZE}],
        "annotations": [ann(0, 1, [closed_ring(square(5, 5, 10))]), ann(1, 1, [closed_ring(square(30, 30, 12))])],
        "categories": [{"id": CATEGORY}],
    }
    predictions = coco_io.gt_as_predictions(gt)
    assert evaluate_ap(gt, predictions) <= 0.5
    repaired, _ = coco_io.subset_gt(gt, None, coco_io.id_offset(gt))
    assert evaluate_ap(repaired, coco_io.gt_as_predictions(repaired)) == pytest.approx(1.0)


def test_gt_as_predictions_reach_ap_100_only_without_the_empty_masks():
    subset, _ = coco_io.subset_gt(source_coco(), None, offset=1)
    assert evaluate_ap(subset, coco_io.gt_as_predictions(subset)) < 1.0  # the zero-area ring cannot match
    clean = coco_io.without_annotations(subset, coco_io.empty_mask_ids(subset))
    assert evaluate_ap(clean, coco_io.gt_as_predictions(clean)) == pytest.approx(1.0)


def test_gt_as_predictions_copy_the_rings():
    subset, _ = coco_io.subset_gt(source_coco(), None, offset=1)
    predictions = coco_io.gt_as_predictions(subset)
    assert len(predictions) == len(subset["annotations"])
    assert all(p["score"] == 1.0 for p in predictions)
    assert predictions[0]["bbox"] == [5.0, 5.0, 10.0, 10.0]
    assert len(subset["annotations"][1]["segmentation"]) == 2 and len(predictions[1]["segmentation"]) == 1  # hole
    predictions[1]["segmentation"][0][0] = -1.0
    assert subset["annotations"][1]["segmentation"][0][0] == 30.0


def test_load_results_accepts_no_prediction():
    coco_gt = coco_io.coco_from_dict(coco_io.subset_gt(source_coco(), None, offset=1)[0])
    empty = coco_io.load_results(coco_gt, [])
    assert empty.getAnnIds() == [] and sorted(empty.getImgIds()) == [1, 2, 3, 4]


def test_coco_objects_work_on_copies():
    data, _ = coco_io.subset_gt(source_coco(), None, offset=1)
    coco = coco_io.coco_from_dict(data)
    coco.dataset["annotations"][0]["segmentation"] = [[]]
    assert data["annotations"][0]["segmentation"] != [[]]
    results = [coco_io.to_result(1, square(5, 5, 10), 0.9, CATEGORY)]
    coco_io.load_results(coco, results)
    assert "id" not in results[0]  # loadRes adds id/area/iscrowd to its own copy only


def test_to_result_closes_the_ring_and_sets_the_bbox():
    result = coco_io.to_result(2, [(1, 2), (11, 2), (11, 7)], 0.5, CATEGORY, theta_deg=12.0)
    assert result["segmentation"] == [[1.0, 2.0, 11.0, 2.0, 11.0, 7.0, 1.0, 2.0]]
    assert result["bbox"] == [1.0, 2.0, 10.0, 5.0]
    assert (result["theta_deg"], result["kappa"]) == (12.0, None)
    with pytest.raises(ValueError, match="at least 3"):
        coco_io.to_result(2, [(1, 2), (3, 4)], 0.5, CATEGORY)


@pytest.fixture
def gt():
    subset, counts = coco_io.subset_gt(source_coco(), None, offset=1)
    return {"info": counts, **subset}


def test_valid_predictions_pass_the_strict_check(gt):
    results = [coco_io.to_result(1, square(5, 5, 10), 0.9, CATEGORY), coco_io.to_result(4, square(0, 0, 64), 0.1, 100)]
    kept, report = coco_io.check_predictions(results, gt)
    assert kept == results
    assert report == {"results": 2, "kept": 2, "dropped": 0, **dict.fromkeys(coco_io.PROBLEMS, 0)}


def broken(problem):
    """A result of image 1 with exactly one problem of coco_io.PROBLEMS."""
    result = coco_io.to_result(1, square(5, 5, 10), 0.9, CATEGORY)
    if problem == "unreadable":
        result["segmentation"] = [[5.0, 5.0, 15.0, 15.0]]  # 4 numbers: a bbox for pycocotools
    elif problem == "too_few_points":
        result["segmentation"] = [[5.0, 5.0, 15.0, 15.0, 5.0, 5.0]]  # [a, b, a], as 0.57% of HiSup's release
    elif problem == "multi_ring":
        result["segmentation"].append(closed_ring(square(7, 7, 2)))
    elif problem == "closing_vertex":
        result["segmentation"] = [result["segmentation"][0][:-2]]
    elif problem == "out_of_frame":
        result = coco_io.to_result(1, square(50, 50, 20), 0.9, CATEGORY)
    elif problem == "bbox_mismatch":
        result["bbox"] = [6.0, 5.0, 10.0, 10.0]
    elif problem == "zero_area":
        result = coco_io.to_result(1, [(5, 5), (10, 10), (15, 15)], 0.9, CATEGORY)
    elif problem == "invalid":
        result = coco_io.to_result(1, [(0, 0), (10, 10), (10, 0), (0, 10)], 0.9, CATEGORY)  # bow tie
    return result


@pytest.mark.parametrize("problem", coco_io.PROBLEMS)
def test_each_problem_is_an_error_in_strict_mode(gt, problem):
    with pytest.raises(ValueError, match=problem):
        coco_io.check_predictions([broken(problem)], gt)


@pytest.mark.parametrize("problem", coco_io.PROBLEMS)
def test_lenient_mode_drops_only_rings_it_cannot_read(gt, problem):
    kept, report = coco_io.check_predictions([broken(problem)], gt, strict=False)
    assert report[problem] == 1 and sum(report[p] for p in coco_io.PROBLEMS) == 1
    assert len(kept) == (0 if problem == "unreadable" else 1)
    assert report["dropped"] == 1 - len(kept)


def test_shapely_readable_leaves_out_rings_of_two_distinct_points():
    good, degenerate = broken("zero_area"), broken("too_few_points")  # 3 collinear points / [a, b, a]
    assert coco_io.shapely_readable([good, degenerate]) == [good]
    assert coco_io.distinct_points(degenerate["segmentation"][0]) == 2


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r.pop("bbox"), "lacks"),
        (lambda r: r.update(image_id=99), "image_id 99"),
        (lambda r: r.update(category_id=1), "category_id 1"),
        (lambda r: r.update(score=float("nan")), "not finite"),
        (lambda r: r.update(segmentation=[[1.0, 2.0, 3.0]]), "flat"),
    ],
)
def test_malformed_results_are_errors_in_both_modes(gt, change, message):
    result = coco_io.to_result(1, square(5, 5, 10), 0.9, CATEGORY)
    change(result)
    for strict in (True, False):
        with pytest.raises(ValueError, match=message):
            coco_io.check_predictions([result], gt, strict=strict)


def test_only_our_outputs_need_theta_and_kappa(gt):
    result = coco_io.to_result(1, square(5, 5, 10), 0.9, CATEGORY)
    del result["theta_deg"], result["kappa"]
    with pytest.raises(ValueError, match="lacks"):
        coco_io.check_predictions([result], gt)
    assert len(coco_io.check_predictions([result], gt, strict=False)[0]) == 1


def test_restrict_to_images():
    results = [{"image_id": i} for i in (1, 2, 3, 2)]
    assert coco_io.restrict_to_images(results, [2]) == [{"image_id": 2}, {"image_id": 2}]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, paths.REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_gt_script(tmp_path, monkeypatch):
    data, out, splits = tmp_path / "data", tmp_path / "outputs", tmp_path / "splits"
    make_fake_crowdai(data)
    monkeypatch.setenv("DENSEPOLY_DATA", str(data))
    monkeypatch.setenv("DENSEPOLY_OUT", str(out))
    write_lines(splits / "crowdai_test.txt", ["t04.png", "t01.png"])
    write_lines(splits / "crowdai_val_clean.txt", ["v20.png", "v21.png"])
    # WHU: both ground truths of the test folder, written as the conversion does (ids from 0, a negative sample)
    whu_test = data / "whu" / "pix2poly_224" / "test"
    whu_test.mkdir(parents=True)
    for name in ("annotation.json", "annotation_p2p.json"):
        (whu_test / name).write_text(json.dumps(source_coco()), encoding="utf-8")
    write_lines(splits / "whu_test.txt", ["img1.png", "img3.png"])

    script = load_script("prepare_gt")
    script.main(["--data", "crowdai", "--split", "test", "val_clean", "val_full", "--splits-dir", str(splits)])
    script.main(["--data", "whu", "--split", "test", "--splits-dir", str(splits)])

    test = read_json(paths.gt_file("crowdai", "test"))
    assert [img["file_name"] for img in test["images"]] == ["t04.png", "t01.png"]
    assert test["info"]["source"] == "crowdai/train/annotation.json" and test["info"]["ann_id_offset"] == 0
    assert {a["image_id"] for a in test["annotations"]} <= {img["id"] for img in test["images"]}
    val_full = read_json(paths.gt_file("crowdai", "val_full"))
    assert len(val_full["images"]) == 4 and val_full["info"]["images_without_annotations"] == 2  # the two copies
    for variant in (None, "p2p"):
        whu = read_json(paths.gt_file("whu", "test", variant))
        assert whu["info"]["variant"] == variant and whu["info"]["dropped_negative_samples"] == 1
        assert [img["id"] for img in whu["images"]] == [1, 3] and [a["id"] for a in whu["annotations"]] == [1, 2]
    with pytest.raises(SystemExit):
        script.main(["--data", "whu", "--split", "val_full"])


def test_gt_file_names_the_variant():
    out = paths.out_root()
    assert paths.gt_file("whu", "test", "p2p") == out / "gt" / "whu" / "test_p2p.json"
    with pytest.raises(ValueError, match="variant"):
        paths.gt_file("crowdai", "test", "p2p")
    assert np.isclose(coco_io.pixel_area([closed_ring(square(5, 5, 10))], SIZE, SIZE), 100, atol=21)
