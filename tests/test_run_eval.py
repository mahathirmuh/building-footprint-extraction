"""One evaluation end to end (densepoly.eval.run_eval and stats; docs/EVALUATION.md §3–5, §6.6) on synthetic data."""

import contextlib
import importlib.util
import io
import pickle

import numpy as np
import pandas as pd
import pytest
from pycocotools.cocoeval import COCOeval

from densepoly import paths
from densepoly.eval import coco_io, run_eval, stats
from densepoly.utils.io import read_json, write_json
from tests.fixtures.coco_toy import make_gt
from tests.fixtures.eval_scene import HOUSES, predictions, write_scene


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Ground truth of crowdai/test and its density file in temporary folders."""
    return write_scene(tmp_path, monkeypatch)


def quiet_stats(evaluator):
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.accumulate()
        evaluator.summarize()
    return [float(s) for s in evaluator.stats]


def segm(gt, results, img_ids=None):
    coco_gt = coco_io.coco_from_dict(gt)
    evaluator = COCOeval(coco_gt, coco_io.load_results(coco_gt, results), "segm")
    if img_ids is not None:
        evaluator.params.imgIds = img_ids
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
    return evaluator


def test_coco_stats_reproduce_pycocotools_exactly():
    gt, results = make_gt(HOUSES), predictions()
    evaluator = segm(gt, results)
    cache = stats.eval_cache(evaluator)
    expected = quiet_stats(evaluator)
    assert list(stats.coco_stats(cache).values()) == expected
    assert stats.coco_ap(cache) == expected[0]
    assert 0 < expected[0] < 1


def test_stratum_stats_equal_cocoeval_on_the_stratum_alone():
    gt, results = make_gt(HOUSES), predictions()
    cache = stats.eval_cache(segm(gt, results))
    for subset in ([1, 2, 6], [4], [3, 5]):
        expected = quiet_stats(segm(gt, results, img_ids=subset))
        assert list(stats.coco_stats(cache, cache.positions(subset)).values()) == expected
        assert stats.coco_ap(cache, cache.positions(subset)) == expected[0]


def test_a_duplicated_image_counts_like_a_copy_of_it():
    """§6.6: resamples with duplicates are what bootstrap AP needs, and COCOeval would drop them."""
    gt, results = make_gt(HOUSES), predictions()
    copy_gt = make_gt({**HOUSES, 7: HOUSES[2]})
    copy_results = results + [{**r, "image_id": 7} for r in results if r["image_id"] == 2]
    expected = quiet_stats(segm(copy_gt, copy_results))[0]
    cache = stats.eval_cache(segm(gt, results))
    assert stats.coco_ap(cache, cache.positions([1, 2, 3, 4, 5, 6, 2])) == expected
    assert stats.coco_ap(cache, cache.positions([1, 2, 3, 4, 5, 6, 2])) != stats.coco_ap(cache)


def test_boundary_stats_equal_the_vendored_accumulate(monkeypatch):
    """Our accumulate of the vendored evalImgs gives what its own accumulate() gives where numpy still has np.float."""
    evaluator = run_eval.boundary_evaluation(make_gt(HOUSES), predictions())
    ours = list(stats.coco_stats(stats.eval_cache(evaluator)).values())
    monkeypatch.setattr(np, "float", float, raising=False)
    assert ours == quiet_stats(evaluator)
    assert ours[0] < list(stats.coco_stats(stats.eval_cache(segm(make_gt(HOUSES), predictions()))).values())[0]


def test_evaluate_writes_the_metrics_and_tables(env):
    gt, splits = env
    results = predictions()
    record = run_eval.evaluate("crowdai", "test", "toy", "ref", results, splits_dir=splits)
    saved = read_json(paths.metrics_file("crowdai", "toy", "ref", "test"))
    assert saved["overall"] == record["overall"] and saved["pred_check"]["mode"] == "strict"
    overall = record["overall"]
    assert overall["n_images"] == 6 and overall["n_pred"] == len(results) and overall["n_gt"] == 13
    assert overall["merge_rate"] == pytest.approx(2 / 13)  # image 6
    assert overall["recall_inst50"] == pytest.approx(10 / 13)  # the miss on image 1, image 6 merged
    assert set(record["per_stratum"]) == {"jarang", "sedang", "padat", "bersentuhan", "tidak_bersentuhan", "kosong"}
    assert record["per_stratum"]["kosong"] == {"IoU": 0.0, "fp_per_image": 1.0, "n_images": 1}
    assert record["per_stratum"]["bersentuhan"]["merge_rate"] == 1.0
    assert record["per_stratum"]["padat"]["recall_inst50"] == pytest.approx(5 / 6)

    table = pd.read_parquet(paths.metrics_artifact("crowdai", "toy", "ref", "test", "per_image.parquet"))
    assert len(table) == 6 and dict(zip(table["image_id"], table["stratum"], strict=True))[6] == "sedang"
    pairs = pd.read_parquet(paths.metrics_artifact("crowdai", "toy", "ref", "test", "pairs.parquet"))
    assert len(pairs) == overall["n_pairs"] == 10
    with paths.metrics_artifact("crowdai", "toy", "ref", "test", "evalimgs.pkl").open("rb") as f:
        caches = pickle.load(f)
    assert stats.coco_stats(caches["segm"]) == record["coco_stats"]["segm"]
    assert stats.coco_stats(caches["boundary"])["AP"] == overall["AP_boundary"]


def test_gt_as_prediction_end_to_end(env):
    """A small §6.1: AP = AP_boundary = C-IoU = 1, PoLiS undefined (all images 0), MTA ≈ 0."""
    gt, splits = env
    record = run_eval.evaluate("crowdai", "test", "gt", "ref", coco_io.gt_as_predictions(gt), splits_dir=splits)
    o = record["overall"]
    assert o["AP"] == pytest.approx(1.0) and o["AP_boundary"] == pytest.approx(1.0)
    assert o["CIoU"] == pytest.approx(1.0) and o["CIoU_p2p"] == pytest.approx(1.0) and o["IoU"] == 1.0
    assert o["PoLiS"] is None and o["PoLiS_valid_images"] == 0
    assert o["MTA"] <= 0.5 and o["MTA_inst"] < 0.01 and o["PoLiS_paired"] == 0.0
    assert (o["recall_inst50"], o["merge_rate"], o["split_rate"]) == (1.0, 0.0, 0.0)


def test_no_prediction_at_all(env):
    gt, splits = env
    o = run_eval.evaluate("crowdai", "test", "none", "ref", [], splits_dir=splits)["overall"]
    assert (o["AP"], o["AP_boundary"], o["recall_inst50"], o["n_pred"]) == (0.0, 0.0, 0.0, 0)
    assert o["CIoU"] == pytest.approx(1 / 6)  # only the empty image scores 1
    assert (o["CIoU_p2p"], o["PoLiS"], o["MTA"], o["PoLiS_paired"]) == (None, None, None, None)


def test_vendor_mta_with_a_pool_matches_the_serial_run(env):
    gt, _ = env
    results = coco_io.check_predictions(predictions(), gt)[0]
    gt_path = paths.gt_file("crowdai", "test")
    serial = run_eval.vendor_metrics(gt_path, gt, results, workers=1)
    pooled = run_eval.vendor_metrics(gt_path, gt, results, workers=2)
    assert pooled["mta_per_image"] == serial["mta_per_image"] and pooled["MTA"] == serial["MTA"]


def test_evaluate_script(env, tmp_path):
    _, splits = env
    pred = paths.pred_file("crowdai", "toy", "seed0", "test")
    write_json(pred, predictions())
    write_json(pred.with_name("test.meta.json"), {"checkpoint": "outputs/runs/crowdai/toy/seed0/best.pt"})
    spec = importlib.util.spec_from_file_location("evaluate", paths.REPO_ROOT / "scripts" / "evaluate.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    script.main(["--data", "crowdai", "--exp", "toy", "--seed", "0", "--split", "test", "--workers", "1",
                 "--splits-dir", str(splits)])  # fmt: skip
    record = read_json(paths.metrics_file("crowdai", "toy", "seed0", "test"))
    assert record["checkpoint"] == "outputs/runs/crowdai/toy/seed0/best.pt"
    assert record["pred_file"] == "preds/crowdai/toy/seed0/test.json"
