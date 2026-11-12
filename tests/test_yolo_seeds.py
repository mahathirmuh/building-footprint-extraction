"""Oriented-box labels from instance maps and the merging of detector seeds into the markers (docs/METHOD.md §8)."""

import numpy as np
import pytest

from densepoly.postproc.markers import label4
from densepoly.postproc.watershed import binary_instances, three_class_instances
from densepoly.postproc.yolo_seeds import instance_boxes, merge_seeds, obb_label_lines, seed_centres
from tests.fixtures.synthetic import instance_map_from_polygons, rect_polygon


def test_the_box_of_an_axis_aligned_house_is_the_house():
    instance_map = np.zeros((40, 60), np.int32)
    instance_map[10:20, 5:35] = 1  # 10 rows × 30 columns
    (box,) = instance_boxes(instance_map)
    assert sorted(map(tuple, np.round(box).astype(int).tolist())) == [(5, 10), (5, 20), (35, 10), (35, 20)]


def test_the_box_follows_a_rotated_house():
    polygon = rect_polygon(50, 50, 40, 16, 30)  # centre, width, height, degrees
    instance_map = instance_map_from_polygons([polygon], 100, 100)
    (box,) = instance_boxes(instance_map)
    sides = sorted(np.linalg.norm(box - np.roll(box, 1, axis=0), axis=1))
    assert sides[0] == pytest.approx(16, abs=2.0) and sides[-1] == pytest.approx(40, abs=2.0)
    edge = box[1] - box[0]
    angle = np.degrees(np.arctan2(edge[1], edge[0])) % 90
    assert min(abs(angle - 30), abs(angle - 30 + 90), abs(angle - 30 - 90)) < 3
    assert np.allclose(box.mean(axis=0), (50, 50), atol=1.0)


def test_label_lines_are_normalised_and_one_per_building():
    instance_map = np.zeros((50, 100), np.int32)
    instance_map[10:20, 10:30] = 3
    instance_map[30:35, 60:70] = 7
    instance_map[0, 0] = 9  # one pixel: under min_area
    lines = obb_label_lines(instance_map, min_area=4)
    assert len(lines) == 2 and all(line.startswith("0 ") for line in lines)
    values = np.array([line.split()[1:] for line in lines], float)
    assert values.shape == (2, 8) and values.min() >= 0 and values.max() <= 1
    xs, ys = values[0, 0::2] * 100, values[0, 1::2] * 50
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == pytest.approx((10, 30, 10, 20), abs=1e-4)
    assert obb_label_lines(np.zeros((8, 8), np.int32)) == []


def test_seed_centres_keep_the_confident_boxes():
    boxes = [[10.0, 5.0, 0.9], [20.0, 6.0, 0.2], [30.0, 7.0, 0.4]]
    assert seed_centres(boxes, 0.4).tolist() == [[10.0, 5.0], [30.0, 7.0]]
    assert seed_centres([], 0.4).shape == (0, 2)
    with_sizes = [[10.0, 5.0, 0.9, 30.0, 12.0, 0.1], [20.0, 6.0, 0.2, 8.0, 8.0, 0.0]]  # as scripts/predict_yolo.py writes
    assert seed_centres(with_sizes, 0.4).tolist() == [[10.0, 5.0]]


def scene():
    """Foreground: a wide blob (two houses fused) with one marker, a house with its own marker, and a house without
    any marker."""
    fg = np.zeros((40, 80), bool)
    fg[5:15, 5:45] = True  # the fused pair
    fg[25:35, 5:20] = True  # a house with a marker
    fg[25:35, 50:65] = True  # a house without a marker
    markers = np.zeros((40, 80), bool)
    markers[7:13, 8:42] = True
    markers[27:33, 8:17] = True
    return fg, label4(markers)


def test_two_seeds_in_one_marker_replace_it():
    fg, markers = scene()
    merged = merge_seeds(markers, fg, np.array([[15.0, 10.0], [35.0, 10.0]]))
    assert merged.max() == 3  # the kept marker and two seeds
    assert merged[10, 15] != merged[10, 35] and merged[10, 15] > 0 and merged[10, 35] > 0
    assert merged[10, 25] == 0  # the old marker between the seeds is gone
    assert (merged[10, 15] == merged).sum() == 13  # a disk of radius 2
    assert (merged[27:33, 8:17] > 0).all()  # the other marker is untouched


def test_one_seed_or_none_leaves_the_marker():
    fg, markers = scene()
    assert np.array_equal(merge_seeds(markers, fg, np.array([[15.0, 10.0]])), markers)
    assert np.array_equal(merge_seeds(markers, fg, np.zeros((0, 2))), markers)


def test_a_seed_is_added_only_where_the_component_has_no_marker():
    fg, markers = scene()
    merged = merge_seeds(markers, fg, np.array([[57.0, 30.0]]))
    assert merged.max() == 3 and merged[30, 57] == 3
    beside = merge_seeds(markers, fg, np.array([[18.5, 30.0]]))  # in the marked house, outside its marker
    assert np.array_equal(beside, markers)


def test_seeds_outside_the_foreground_or_the_tile_are_ignored():
    fg, markers = scene()
    outside = np.array([[70.0, 3.0], [-5.0, 10.0], [200.0, 10.0], [10.0, 99.0]])
    assert np.array_equal(merge_seeds(markers, fg, outside), markers)


def test_a_seed_is_cut_to_the_foreground_and_never_paints_a_kept_marker():
    fg = np.zeros((20, 40), bool)
    fg[5:8, 5:15] = True  # a thin unmarked strip, 3 pixels high
    fg[5:15, 20:35] = True
    markers = label4(np.pad(np.ones((6, 9), bool), ((7, 7), (22, 9))))
    merged = merge_seeds(markers, fg, np.array([[10.0, 6.0]]))
    seed = merged == merged[6, 10]
    assert seed.sum() == 11 and not (seed & ~fg).any()  # the disk loses the pixels outside the strip
    assert np.array_equal(merged > 0, (markers > 0) | seed)


def test_seeds_split_a_fused_pair_in_the_watershed():
    """Two houses that the network sees as one blob: one instance without seeds, two with a seed in each."""
    p_b = np.zeros((30, 60), np.float32)
    p_b[8:22, 6:54] = 0.95
    p_bg, p_k = 1.0 - p_b, np.zeros_like(p_b)
    plain = three_class_instances(p_bg, p_b, p_k, 0.4, 8, 10)
    assert plain.count == 1
    seeded = three_class_instances(p_bg, p_b, p_k, 0.4, 8, 10, seeds=np.array([[18.0, 15.0], [42.0, 15.0]]))
    assert seeded.count == 2
    assert seeded.instance_map[15, 18] != seeded.instance_map[15, 42]
    assert np.array_equal(seeded.instance_map > 0, plain.instance_map > 0)  # the foreground is the same


def test_seeds_split_a_binary_component_at_its_neck():
    """Two houses joined by a 2-pixel bridge: one component without seeds; with a seed in each, cut at the bridge."""
    probability = np.zeros((30, 70), np.float32)
    probability[5:25, 5:30] = 0.9  # 25 columns
    probability[5:25, 38:65] = 0.9  # 27 columns
    probability[14:16, 30:38] = 0.9  # the bridge
    plain = binary_instances(probability, 10)
    assert plain.count == 1
    seeded = binary_instances(probability, 10, seeds=np.array([[17.0, 15.0], [51.0, 15.0]]))
    assert seeded.count == 2
    left, right = seeded.instance_map[15, 17], seeded.instance_map[15, 51]
    assert left != right
    assert (seeded.instance_map[5:25, 5:30] == left).all() and (seeded.instance_map[5:25, 38:65] == right).all()
    assert np.array_equal(seeded.instance_map > 0, plain.instance_map > 0)
    assert seeded.scores == pytest.approx([0.9, 0.9])


def test_binary_components_with_one_seed_or_none_are_unchanged():
    probability = np.zeros((30, 70), np.float32)
    probability[5:25, 5:30] = 0.9
    probability[5:25, 40:65] = 0.8
    plain = binary_instances(probability, 10)
    for seeds in (np.zeros((0, 2)), np.array([[17.0, 15.0]]), np.array([[17.0, 15.0], [52.0, 15.0], [35.0, 2.0]])):
        seeded = binary_instances(probability, 10, seeds=seeds)
        assert np.array_equal(seeded.instance_map, plain.instance_map)
        assert seeded.scores == pytest.approx(plain.scores)
