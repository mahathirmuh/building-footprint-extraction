"""Oracle post-processing on synthetic ground truth (densepoly.postproc.oracle; docs/EVALUATION.md §6.3)."""

import numpy as np
import pytest

from densepoly.eval import coco_io
from densepoly.eval.orientation import polygon_theta_deg
from densepoly.postproc import oracle
from densepoly.postproc.polygonize import PolygonParams
from tests.fixtures.synthetic import make_scene, to_coco


@pytest.fixture(scope="module")
def gt():
    return to_coco([make_scene(seed) for seed in range(3)])


def test_round_trip_is_exact(gt):
    ious = oracle.round_trip_ious(gt)
    assert len(ious) == len(gt["annotations"]) and ious.min() == 1.0


def test_instance_maps_are_the_training_labels(gt):
    scenes = [make_scene(seed) for seed in range(3)]
    for (image, raster), scene in zip(oracle.instance_maps(gt), scenes, strict=True):
        assert np.array_equal(raster.instance_map, scene.instance_map), image["id"]


@pytest.mark.parametrize("mode", ["raw", "mar", "theta"])
def test_instance_oracle_gives_one_valid_polygon_per_house(gt, mode):
    results = oracle.instance_predictions(gt, PolygonParams(mode))
    kept, _ = coco_io.check_predictions(results, gt, strict=True)
    assert len(kept) == len(gt["annotations"]) and all(r["score"] == 1.0 for r in kept)
    if mode == "theta":  # houses come in annotation order within each image
        expected = [polygon_theta_deg(a["segmentation"][0]) for a in gt["annotations"]]
        assert [r["theta_deg"] for r in kept] == pytest.approx(expected)


def test_watershed_oracle_finds_every_house(gt):
    results = oracle.watershed_predictions(gt, PolygonParams("raw"), 2, 0.4, 8, 10)
    kept, _ = coco_io.check_predictions(results, gt, strict=True)
    per_image = [sum(r["image_id"] == img["id"] for r in kept) for img in gt["images"]]
    assert per_image == [sum(a["image_id"] == img["id"] for a in gt["annotations"]) for img in gt["images"]]
    assert all(r["score"] == pytest.approx(1.0) for r in kept)
