"""Marching-squares exterior rings (densepoly.postproc.contours; docs/METHOD.md §6.1, docs/EVALUATION.md §6.3 (a))."""

import numpy as np
import pytest
from scipy import ndimage as ndi

from densepoly.postproc.contours import mask_to_exterior, shoelace_area
from tests.fixtures.synthetic import l_polygon, polygon_mask, rect_polygon

FOUR = ndi.generate_binary_structure(2, 1)


def test_single_pixel_gives_the_diamond_through_its_edge_midpoints():
    mask = np.zeros((6, 6), bool)
    mask[2, 3] = True
    ring = mask_to_exterior(mask)
    assert sorted(map(tuple, ring)) == [(3.0, 2.5), (3.5, 2.0), (3.5, 3.0), (4.0, 2.5)]


def test_square_loses_only_its_four_corners():
    mask = np.zeros((20, 20), bool)
    mask[5:15, 7:17] = True
    ring = mask_to_exterior(mask)
    assert abs(shoelace_area(ring)) == pytest.approx(100 - 4 / 8)
    assert (ring.min(axis=0).tolist(), ring.max(axis=0).tolist()) == ([7.0, 5.0], [17.0, 15.0])
    assert not np.array_equal(ring[0], ring[-1])  # no repeated closing vertex


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Instances are 4-connected (docs/METHOD.md §6.2), so the round trip is checked on 4-connected masks."""
    labels, n = ndi.label(mask, structure=FOUR)
    if n == 0:
        return mask
    return labels == np.argmax(np.bincount(labels.ravel())[1:]) + 1


def test_round_trip_through_pycocotools_is_exact_on_random_shapes():
    """§6.3 (a) on synthetic shapes: mask -> ring -> pycocotools raster gives the same mask, pixel for pixel."""
    rng = np.random.default_rng(0)
    size, checked = 64, 0
    for i in range(300):
        cx, cy = rng.uniform(-4, size + 4, 2)  # some shapes cross the tile border
        w, h = rng.uniform(1.5, 30, 2)
        angle = rng.uniform(0, 90)
        if i % 3 == 2:
            polygon = l_polygon(cx, cy, max(w, 6), max(h, 6), min(w, h, 5) / 2 + 1, angle)
        else:
            polygon = rect_polygon(cx, cy, w, h, angle)
        mask = largest_component(polygon_mask(polygon, size, size))
        if not mask.any():
            continue
        assert np.array_equal(polygon_mask(mask_to_exterior(mask), size, size), mask), i
        checked += 1
    assert checked > 250


def test_holes_are_ignored():
    mask = np.zeros((12, 12), bool)
    mask[2:10, 2:10] = True
    mask[5:7, 5:7] = False
    filled = ndi.binary_fill_holes(mask)
    assert np.array_equal(polygon_mask(mask_to_exterior(mask), 12, 12), filled)


def test_diagonal_pixels_stay_apart():
    """Two pixels touching only at a corner are two 4-connected parts; the ring follows one of them."""
    mask = np.zeros((5, 5), bool)
    mask[1, 1] = mask[2, 2] = True
    assert abs(shoelace_area(mask_to_exterior(mask))) == pytest.approx(0.5)


def test_empty_mask_raises():
    with pytest.raises(ValueError, match="empty"):
        mask_to_exterior(np.zeros((4, 4), bool))
