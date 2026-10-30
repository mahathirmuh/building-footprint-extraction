"""Classical baseline B0 on synthetic MBI maps (densepoly.classical.b0; docs/METHOD.md §8)."""

import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage.morphology import h_maxima

from densepoly.classical.b0 import B0Params, foreground, h_maxima_markers, instances, threshold
from densepoly.postproc.markers import label4
from densepoly.priors.mbi import brightness, mbi
from tests.fixtures.synthetic import make_scene, polygon_mask, rect_polygon


def bright(*polygons, size=96, level=0.8, ground=0.05):
    """An MBI-like map: bright polygons on a dark ground."""
    out = np.full((size, size), ground)
    for polygon in polygons:
        out[polygon_mask(polygon, size, size)] = level
    return out


def test_threshold_is_otsu_with_a_lower_bound():
    two_levels = bright(rect_polygon(48, 48, 30, 20))
    assert 0.05 < threshold(two_levels, 0.15) < 0.8
    noise = np.random.default_rng(0).uniform(0, 0.1, (64, 64))  # no building: Otsu would still split it
    assert threshold(noise, 0.15) == 0.15
    assert threshold(np.zeros((8, 8)), 0.15) == 0.15


def test_foreground_opens_drops_specks_and_fills_holes():
    mbi_map = bright(rect_polygon(30, 30, 30, 20), rect_polygon(75, 75, 4, 4))  # a house and a 16 px speck
    mbi_map[28:32, 28:32] = 0.05  # a dark hole inside the house
    fg = foreground(mbi_map, B0Params())
    assert fg[28:32, 28:32].all() and not fg[73:77, 73:77].any()
    assert label4(fg).max() == 1


@pytest.mark.parametrize("angle", [0, 10, 45, 80])
def test_a_long_house_has_one_marker_at_any_angle(angle):
    """Along its ridge the distance wavers by a fraction of a pixel: skimage's h_maxima keeps only the top pixels, a
    row of separate markers, whereas the regional maxima of the h-maxima transform give one."""
    mbi_map = bright(rect_polygon(48, 48, 50, 14, angle))
    fg = foreground(mbi_map, B0Params())
    distance = ndi.distance_transform_edt(fg)
    if angle == 80:
        assert label4(h_maxima(distance, 2.0) > 0).max() > 1
    assert h_maxima_markers(distance, 2.0, fg).max() == 1
    assert instances(mbi_map, B0Params()).count == 1


def test_two_overlapping_disks_split_at_their_neck():
    yy, xx = np.mgrid[:96, :96]
    mbi_map = np.where(((xx - 30) ** 2 + (yy - 48) ** 2 < 15**2) | ((xx - 58) ** 2 + (yy - 48) ** 2 < 15**2), 0.8, 0.05)
    inst = instances(mbi_map, B0Params())
    assert inst.count == 2 and np.all(inst.scores == 1)
    sizes = np.bincount(inst.instance_map.ravel())[1:]
    assert abs(int(sizes[0]) - int(sizes[1])) <= 0.05 * sizes.sum()


def test_separate_houses_are_separate_instances():
    mbi_map = bright(rect_polygon(25, 25, 24, 16, 10), rect_polygon(70, 30, 20, 20), rect_polygon(48, 75, 40, 12, 80))
    assert instances(mbi_map, B0Params()).count == 3


@pytest.mark.parametrize("t_min", [0.05, 0.15, 0.25])
def test_b0_on_synthetic_scenes_finds_most_isolated_houses(t_min):
    """End to end from RGB: the MBI of the image (scaled by its own 99.5th percentile), then B0."""
    found = total = 0
    for seed in range(3):
        scene = make_scene(seed)
        raw = mbi(brightness(scene.image), list(range(11, 82, 10)))
        inst = instances(np.clip(raw / np.percentile(raw, 99.5), 0, 1), B0Params(t_min=t_min)).instance_map
        for b in scene.buildings:
            if b.kind != "rect":
                continue
            house = scene.instance_map == b.id
            ids, counts = np.unique(inst[house], return_counts=True)
            best = ids[np.argmax(counts)]
            total += 1
            found += best > 0 and (house & (inst == best)).sum() / (house | (inst == best)).sum() >= 0.5
    assert found >= 0.7 * total, (found, total)
