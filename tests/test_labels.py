"""Orientation, boundary weight and the derived targets (docs/METHOD.md §3.3-3.4) on cases with a known answer."""

import numpy as np
import pytest

from densepoly.labels.boundary_weight import boundary_mask, edge_weight
from densepoly.labels.contact import class_map
from densepoly.labels.orientation import orientation_targets
from densepoly.labels.targets import derive_targets
from tests.fixtures.synthetic import instance_map_from_polygons, make_scene, rect_polygon


def rotated_house(angle, size=128):
    return instance_map_from_polygons([rect_polygon(size / 2, size / 2, 60, 30, angle)], size, size)


@pytest.mark.parametrize("angle", [0, 10, 30, 45, 60, 80, 89])
def test_theta_of_a_rotated_house_is_its_angle(angle):
    (theta,) = orientation_targets(rotated_house(angle), 20).theta.values()
    error = (np.degrees(theta) - angle + 45) % 90 - 45  # difference modulo 90°
    assert abs(error) <= 1.0


@pytest.mark.parametrize(
    "op, sign", [(np.fliplr, -1), (np.flipud, -1), (np.transpose, -1), (np.rot90, 1)], ids=lambda v: str(v)
)
def test_sin4_changes_sign_under_flips_and_transpose_but_not_under_rot90(op, sign):
    house = rotated_house(30)
    (base,) = orientation_targets(house, 20).theta.values()
    (moved,) = orientation_targets(np.ascontiguousarray(op(house)), 20).theta.values()
    assert np.sin(4 * moved) == pytest.approx(sign * np.sin(4 * base), abs=1e-4)
    assert np.cos(4 * moved) == pytest.approx(np.cos(4 * base), abs=1e-4)


def test_orientation_targets_cover_only_instances_of_at_least_the_minimum_area():
    labels = np.zeros((40, 40), np.uint16)
    labels[2:6, 2:6] = 1  # 16 px < 20
    labels[10:30, 10:20] = 2  # axis-aligned: θ = 0
    targets = orientation_targets(labels, 20)
    assert set(targets.theta) == {2} and np.array_equal(targets.valid, labels == 2)
    np.testing.assert_allclose(targets.cos4[labels == 2], 1.0, atol=1e-6)
    assert not targets.sin4[~targets.valid].any() and not targets.cos4[~targets.valid].any()


@pytest.mark.parametrize("direction, cos4", [("horizontal", 1.0), ("vertical", 1.0), ("diagonal", -1.0)])
def test_a_one_pixel_wide_building_keeps_its_direction(direction, cos4):
    """minAreaRect of collinear pixels has a zero-length side; θ must come from the other side."""
    labels = np.zeros((64, 64), np.uint16)
    steps = np.arange(10, 40)
    rows, cols = {"horizontal": (20, steps), "vertical": (steps, 20), "diagonal": (steps, steps)}[direction]
    labels[rows, cols] = 1
    targets = orientation_targets(labels, 20)
    np.testing.assert_allclose(targets.cos4[labels == 1], cos4, atol=1e-6)


def test_boundary_covers_both_sides_of_every_edge():
    labels = np.zeros((10, 10), np.uint16)
    labels[3:6, 3:6] = 1
    expected = np.zeros((10, 10), bool)
    expected[3:6, 3:6] = True
    expected[4, 4] = False  # the centre of the 3 × 3 house has no different 4-neighbour
    expected[2, 3:6] = expected[6, 3:6] = expected[3:6, 2] = expected[3:6, 6] = True  # outside, 4-neighbours only
    assert np.array_equal(boundary_mask(labels), expected)


def test_shared_walls_are_boundaries_too():
    labels = np.zeros((20, 20), np.uint16)
    labels[5:15, 2:10] = 1
    labels[5:15, 10:18] = 2
    assert boundary_mask(labels)[5:15, 9:11].all()


def test_edge_weight_is_one_plus_alpha_on_the_boundary_and_decays_with_distance():
    labels = np.zeros((64, 64), np.uint16)
    labels[20:40, 20:40] = 1
    weight = edge_weight(labels, alpha=5.0, sigma=3.0)
    np.testing.assert_allclose(weight[boundary_mask(labels)], 6.0)
    assert weight.max() == pytest.approx(6.0)
    assert weight[30, 30] == pytest.approx(1 + 5 * np.exp(-(9**2) / (2 * 3.0**2)), rel=1e-5)  # 9 px from row 39
    assert weight[0, 0] == pytest.approx(1.0, abs=1e-6)
    assert np.array_equal(edge_weight(np.zeros((8, 8), np.uint16), 5.0, 3.0), np.ones((8, 8), np.float32))


@pytest.mark.parametrize("seed", range(3))
def test_derived_targets_are_consistent(seed):
    scene = make_scene(seed)
    targets = derive_targets(scene.instance_map, contact_radius=2, theta_min_area=20, alpha=5.0, sigma=3.0)
    assert np.array_equal(targets.classes, class_map(scene.instance_map, 2))
    assert not (targets.valid & (scene.instance_map == 0)).any()  # θ only on buildings
    for b in scene.buildings:  # every synthetic house is large enough and gets its own angle
        own = scene.instance_map == b.id
        assert targets.valid[own].all()
        np.testing.assert_allclose(targets.sin4[own], np.sin(np.radians(4 * b.theta_deg)), atol=0.06)
    assert targets.weight.min() >= 1.0 and targets.weight.dtype == np.float32
