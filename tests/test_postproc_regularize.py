"""Polygon regularisation (densepoly.postproc.regularize; docs/METHOD.md §7, PLAN.md Phase 4 "Selesai jika").

The shapes, angles and edge noise follow reference/prototypes/regularize_proto.py; masks are rasterised with pycocotools
like the ground truth, and θ is the minAreaRect θ of the mask (mode mar)."""

import numpy as np
import pytest
from scipy import ndimage as ndi
from shapely import affinity
from shapely.geometry import Polygon, box

from densepoly.labels.orientation import pixel_theta
from densepoly.postproc.contours import mask_to_exterior
from densepoly.postproc.regularize import (
    RegularizeParams,
    iou,
    moment_box,
    outline_polygon,
    regularize,
    select,
    vertices,
)
from tests.fixtures.synthetic import polygon_mask, rect_polygon

SIZE = 160
ANGLES = (0, 10, 20, 33, 45, 58, 70, 85)
SHAPES = {  # name: (polygon, vertices expected without noise)
    "box": (Polygon(rect_polygon(80, 80, 70, 40)), 4),
    "L": (Polygon([(40, 40), (110, 40), (110, 70), (70, 70), (70, 120), (40, 120)]), 6),
    "T": (Polygon([(30, 50), (130, 50), (130, 80), (95, 80), (95, 125), (65, 125), (65, 80), (30, 80)]), 8),
    "oblique wall": (Polygon([(40, 40), (120, 40), (120, 85), (95, 115), (40, 115)]), 5),
}
SMALL_BOX = Polygon(rect_polygon(80, 80, 12, 9))
FOUR = ndi.generate_binary_structure(2, 1)


def mask_of(polygon: Polygon) -> np.ndarray:
    return polygon_mask(np.asarray(polygon.exterior.coords)[:-1], SIZE, SIZE)


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, _ = ndi.label(mask, structure=FOUR)
    return labels == np.argmax(np.bincount(labels.ravel())[1:]) + 1


def jagged(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Flip a quarter of the pixels on the 3 × 3 morphological gradient, then keep the largest 4-connected part (an
    instance is 4-connected, docs/METHOD.md §6.2)."""
    edge = ndi.binary_dilation(mask, np.ones((3, 3))) & ~ndi.binary_erosion(mask, np.ones((3, 3)))
    flip = edge & (rng.random(mask.shape) < 0.25)
    return largest_component(mask ^ flip)


def regularize_mar(mask: np.ndarray):
    rows, cols = np.nonzero(mask)
    return regularize(mask, pixel_theta(rows, cols))


@pytest.mark.parametrize("name", SHAPES)
def test_clean_shapes_keep_their_vertices_at_eight_angles(name):
    base, expected = SHAPES[name]
    for angle in ANGLES:
        truth = affinity.rotate(base, angle, origin=(80, 80))
        result = regularize_mar(mask_of(truth))
        assert len(vertices(result.polygon)) == expected, (angle, result.path)
        assert iou(result.polygon, truth) >= 0.95, (angle, result.path)


def test_small_box_with_its_true_theta_is_a_box_no_worse_than_its_raster():
    """A 12 × 9 box. Its raster alone reaches only IoU ±0.89 with the polygon when edges fall on pixel centres, so the
    check is 4 vertices and no less IoU than the raster's own contour. The true θ is given: on so few pixels the
    minAreaRect θ of mode mar misses the 10° and 85° boxes by 5-10° (it returns 0°), a limit of mar that the θ head
    (N1) addresses, not of the regularisation."""
    for angle in ANGLES:
        truth = affinity.rotate(SMALL_BOX, angle, origin=(80, 80))
        mask = mask_of(truth)
        result = regularize(mask, np.radians(angle % 90))
        assert (len(vertices(result.polygon)), result.path) == (4, "box"), angle
        assert iou(result.polygon, truth) >= iou(Polygon(mask_to_exterior(mask)), truth) - 0.01, angle


def test_noisy_box_stays_a_box_and_regularisation_beats_the_contour():
    rng = np.random.default_rng(1)
    for name, (base, _) in SHAPES.items():
        gains = []
        for angle in ANGLES:
            truth = affinity.rotate(base, angle, origin=(80, 80))
            mask = jagged(mask_of(truth), rng)
            result = regularize_mar(mask)
            if name == "box":
                assert len(vertices(result.polygon)) == 4, (angle, result.path)
            gains.append(iou(result.polygon, truth) - iou(Polygon(mask_to_exterior(mask)), truth))
        assert np.mean(gains) >= 0, (name, gains)


def test_moment_box_of_a_pixel_rectangle_is_that_rectangle():
    mask = np.zeros((30, 40), bool)
    mask[5:15, 7:27] = True
    rows, cols = np.nonzero(mask)
    assert moment_box(rows, cols, 0.0).bounds == pytest.approx((7, 5, 27, 15))


def test_moment_box_shrugs_off_a_bump_that_stretches_a_bounding_box():
    mask = np.zeros((40, 60), bool)
    mask[10:30, 10:40] = True
    mask[19:21, 40:46] = True  # a 6 px spike, e.g. a wrongly segmented antenna
    rows, cols = np.nonzero(mask)
    truth = box(10, 10, 40, 30)
    bounding = box(cols.min(), rows.min(), cols.max() + 1, rows.max() + 1)
    assert iou(moment_box(rows, cols, 0.0), truth) > 0.95 > 0.85 > iou(bounding, truth)


def test_an_invalid_snap_result_falls_back_to_the_dp_polygon_then_to_the_contour():
    outline = box(0, 0, 10, 10)
    bow_tie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])
    dp = Polygon([(0, 0), (10, 0), (10, 10), (0, 9.5)])
    chosen = select([("snap", bow_tie), ("dp", dp)], outline)
    assert (chosen.path, chosen.fallback, chosen.polygon.is_valid) == ("dp", True, True)
    chosen = select([("snap", bow_tie), ("dp", box(20, 20, 30, 30))], outline)  # valid but far from the contour
    assert chosen.path == "contour" and chosen.polygon.equals(outline)


def test_any_mask_and_any_theta_give_a_valid_polygon_close_to_the_contour():
    """Wrong θ, tiny masks and one-pixel lines still end in a valid polygon with IoU ≥ 0.7 to the contour (§7.3)."""
    rng = np.random.default_rng(2)
    masks = [
        mask_of(Polygon(rect_polygon(80, 80, w, h, a))) for w, h, a in rng.uniform([1, 1, 0], [40, 40, 90], (40, 3))
    ]
    masks += [
        jagged(mask_of(affinity.rotate(SHAPES[n][0], a, origin=(80, 80))), rng) for n in ("L", "T") for a in ANGLES
    ]
    single = np.zeros((SIZE, SIZE), bool)
    single[5, 5] = True
    line = np.zeros((SIZE, SIZE), bool)
    line[50, 20:70] = True
    masks += [single, line]
    params = RegularizeParams(rect_iou=0.99)  # force the snap path more often
    for i, mask in enumerate(masks):
        mask = largest_component(mask) if mask.any() else mask
        if not mask.any():
            continue
        for theta in (0.0, rng.uniform(0, np.pi / 2)):
            result = regularize(mask, theta, params)
            outline = outline_polygon(mask_to_exterior(mask))
            assert result.polygon.is_valid and result.polygon.area > 0, (i, theta, result.path)
            assert iou(result.polygon, outline) >= 0.7, (i, theta, result.path)
