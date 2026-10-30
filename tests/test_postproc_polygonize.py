"""Polygons of an instance map (densepoly.postproc.polygonize; docs/METHOD.md §6.4, §7, docs/EVALUATION.md §1)."""

import numpy as np
import pytest
from shapely.geometry import Polygon

from densepoly.eval import coco_io
from densepoly.postproc.polygonize import PolygonParams, head_theta, polygonize, tile_ring, to_results
from densepoly.postproc.regularize import iou
from tests.fixtures.synthetic import make_scene, polygon_mask, rect_polygon, to_coco

CATEGORY = 100


@pytest.mark.parametrize("mode", ["raw", "mar", "theta"])
def test_scene_polygons_pass_the_strict_check_and_fit_their_houses(mode):
    scenes = [make_scene(seed) for seed in range(3)]
    gt = to_coco(scenes, CATEGORY)
    results = []
    for image_id, scene in enumerate(scenes, start=1):
        truth = {b.id: Polygon(b.polygon) for b in scene.buildings}
        theta = np.radians([b.theta_deg for b in scene.buildings])
        polygons = polygonize(scene.instance_map, PolygonParams(mode), theta=theta)
        assert [p.instance for p in polygons] == sorted(truth)
        for p in polygons:
            assert iou(Polygon(p.points), truth[p.instance]) >= 0.85, (mode, image_id, p.instance, p.path)
            assert (p.theta_deg is None) == (mode == "raw")
        results += to_results(polygons, np.ones(len(polygons)), image_id, CATEGORY)
    kept, counts = coco_io.check_predictions(results, gt, strict=True)  # raises on any problem of §1
    assert counts["kept"] == len(results) == sum(len(s.buildings) for s in scenes)
    assert all(np.allclose(r["segmentation"][0], np.round(r["segmentation"][0], 2)) for r in kept)


def test_mar_turns_a_rotated_house_into_a_box():
    instance_map = polygon_mask(rect_polygon(40, 40, 30, 16, 25), 80, 80).astype(np.int32)
    (p,) = polygonize(instance_map, PolygonParams("mar"))
    assert (len(p.points), p.path, p.fallback, p.kappa) == (4, "box", False, None)
    assert p.theta_deg == pytest.approx(25, abs=1)


def test_a_box_sticking_out_of_the_tile_is_clipped():
    instance_map = np.zeros((40, 40), np.int32)
    instance_map[0:12, 0:12] = 1
    instance_map[0:3, 12:20] = 1  # a strip that makes the moment box wider than the tile corner allows
    (p,) = polygonize(instance_map, PolygonParams("theta"), theta=np.radians([30.0]))
    assert p.points.min() >= 0 and p.points[:, 0].max() <= 40 and p.points[:, 1].max() <= 40


def test_head_theta_is_the_weighted_circular_mean_and_kappa_its_agreement():
    rng = np.random.default_rng(0)
    instance_map = np.zeros((20, 40), np.int32)
    instance_map[:, :20], instance_map[:, 20:] = 1, 2
    angle = np.where(instance_map == 1, np.radians(30), np.radians(89))
    angle = angle + np.radians(rng.normal(0, 3, angle.shape))
    angle[:, 30:] = np.radians(1)  # half of instance 2 at 1°: next to 89° across the 0°/90° wrap
    sin4, cos4 = 0.8 * np.sin(4 * angle), 0.8 * np.cos(4 * angle)  # tanh outputs need not be unit vectors
    theta, kappa = head_theta(instance_map, sin4, cos4, np.ones_like(sin4))
    assert np.degrees(theta[0]) == pytest.approx(30, abs=0.5) and kappa[0] > 0.95
    assert min(np.degrees(theta[1]), 90 - np.degrees(theta[1])) < 1.5  # 0°, not the 45° of a naive mean
    mixed = np.where(np.arange(40) % 2, np.radians(0.0), np.radians(45.0))[None, :].repeat(20, 0)  # 4θ: 0° vs 180°
    _, kappa = head_theta(instance_map, np.sin(4 * mixed), np.cos(4 * mixed), np.ones_like(mixed))
    assert kappa.max() < 1e-6


def test_tile_ring_rejects_what_rounding_breaks():
    sliver = Polygon([(0, 0), (10, 0.001), (10, 0.004), (0, 0.002)])
    assert sliver.is_valid and tile_ring(sliver, 20, 20) is None
    points = tile_ring(Polygon([(0.123, 0.456), (5.789, 0.1), (5.0, 4.4449)]), 20, 20)
    assert sorted(map(tuple, points.tolist())) == [(0.12, 0.46), (5.0, 4.44), (5.79, 0.1)]  # the clip may turn the ring


def test_theta_mode_needs_theta_and_unknown_modes_fail():
    with pytest.raises(ValueError, match="θ per instance"):
        polygonize(np.ones((4, 4), np.int32), PolygonParams("theta"))
    with pytest.raises(ValueError, match="unknown polygon mode"):
        PolygonParams("box")


def test_results_carry_score_theta_kappa_and_fallback():
    instance_map = np.zeros((30, 30), np.int32)
    instance_map[2:12, 2:12], instance_map[15:25, 15:28] = 1, 2
    polygons = polygonize(instance_map, PolygonParams("theta"), theta=np.zeros(2), kappa=np.array([0.9, 0.4]))
    results = to_results(polygons, np.array([0.8, 0.3]), 7, CATEGORY)
    assert [(r["image_id"], r["score"], r["theta_deg"], r["kappa"], r["fallback"]) for r in results] == [
        (7, 0.8, 0.0, 0.9, False),
        (7, 0.3, 0.0, 0.4, False),
    ]
