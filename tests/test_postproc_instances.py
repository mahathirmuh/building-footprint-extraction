"""Markers, watershed and instances of a prediction (postproc.markers and .watershed; docs/METHOD.md §6.2-6.3)."""

import numpy as np

from densepoly.labels.contact import class_map
from densepoly.postproc.markers import drop_small, foreground, markers
from densepoly.postproc.watershed import (
    binary_instances,
    mean_per_instance,
    split_parts,
    three_class_instances,
    watershed_instances,
)
from tests.fixtures.synthetic import make_scene


def one_hot(classes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perfect probabilities (P_bg, P_b, P_k) of a three-class label."""
    return tuple((classes == c).astype(np.float32) for c in (0, 1, 2))


def same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    """Two instance maps with the same instances, whatever their numbering."""
    pairs = np.unique(np.stack([a.ravel(), b.ravel()]), axis=1)
    return len(pairs[0]) == len(np.unique(a)) == len(np.unique(b)) and np.array_equal(a == 0, b == 0)


def test_foreground_is_strictly_above_one_half():
    assert foreground(np.array([0.49, 0.5, 0.51])).tolist() == [True, False, False]


def test_markers_are_cut_to_the_foreground_before_labelling():
    """Two houses 1 px apart where P_b − P_k exceeds τ in the gap but F does not hold: two markers, not one."""
    p_bg = np.ones((10, 21), np.float32)
    p_b = np.zeros_like(p_bg)
    p_bg[2:8, 2:10] = p_bg[2:8, 11:19] = 0.0
    p_b[2:8, 2:10] = p_b[2:8, 11:19] = 1.0
    p_bg[2:8, 10], p_b[2:8, 10] = 0.55, 0.45  # gap: background for F, marker for P_b − P_k > 0.4
    fg = foreground(p_bg)
    assert markers(p_b, np.zeros_like(p_b), fg, 0.4, 1).max() == 2
    inst = three_class_instances(p_bg, p_b, np.zeros_like(p_b), 0.4, 1, 1)
    assert inst.count == 2 and (inst.instance_map[2:8, 10] == 0).all()


def test_small_markers_go_but_their_component_stays_an_instance():
    p_bg = np.ones((12, 12), np.float32)
    p_b, p_k = np.zeros_like(p_bg), np.zeros_like(p_bg)
    p_bg[2:10, 2:10] = 0.0
    p_k[2:10, 2:10] = 1.0  # all contact ...
    p_b[5:7, 5:7], p_k[5:7, 5:7] = 1.0, 0.0  # ... but a 4 px core
    fg = foreground(p_bg)
    assert markers(p_b, p_k, fg, 0.4, 8).max() == 0  # m_min = 8 drops the core
    inst = three_class_instances(p_bg, p_b, p_k, 0.4, 8, 10)
    assert inst.count == 1 and (inst.instance_map > 0).sum() == 64  # unreached by any marker, still one house


def test_contact_band_splits_two_touching_houses_along_their_wall():
    truth = np.zeros((20, 30), np.int32)
    truth[4:16, 3:15], truth[4:16, 15:27] = 1, 2  # shared wall between columns 14 and 15
    inst = three_class_instances(*one_hot(class_map(truth.astype(np.uint16), 2)), 0.4, 8, 10)
    assert inst.count == 2 and same_partition(inst.instance_map, truth)
    assert np.allclose(inst.scores, 1.0)


def test_unreached_foreground_and_min_area():
    fg = np.zeros((10, 20), bool)
    fg[1:6, 1:6] = True  # 25 px with a marker
    fg[1:4, 10:13] = True  # 9 px, no marker: an instance of its own unless smaller than A_min
    marker_labels = np.zeros((10, 20), np.int32)
    marker_labels[3, 3] = 1
    flat = np.zeros((10, 20))
    assert watershed_instances(flat, marker_labels, fg, 5).max() == 2
    assert watershed_instances(flat, marker_labels, fg, 10).max() == 1
    assert watershed_instances(flat, np.zeros_like(marker_labels), fg, 1).max() == 2  # no marker at all


def test_split_parts_cuts_instances_at_diagonal_contacts_and_drops_small_parts():
    instance_map = np.zeros((8, 8), np.int32)
    instance_map[0:3, 0:3] = 1
    instance_map[3:5, 3:5] = 1  # touches the first block only at a corner
    instance_map[6, 6] = 2
    parts = split_parts(instance_map, 4)
    assert parts.max() == 2 and (parts == 0).sum() == 64 - 9 - 4


def test_scores_are_means_per_instance():
    instance_map = np.array([[1, 1, 0], [2, 2, 2]], np.int32)
    values = np.array([[0.2, 0.4, 0.9], [1.0, 0.5, 0.0]])
    assert np.allclose(mean_per_instance(instance_map, values), [0.3, 0.5])


def test_binary_instances_are_4_connected_components():
    prob = np.zeros((6, 6), np.float32)
    prob[0:2, 0:2] = 0.9
    prob[2:4, 2:4] = 0.7  # diagonal contact only: two instances
    prob[5, 5] = 0.8  # one pixel, below A_min
    inst = binary_instances(prob, 2)
    assert inst.count == 2 and np.allclose(sorted(inst.scores), [0.7, 0.9])


def test_drop_small_keeps_order():
    labels = np.array([[3, 3, 0, 1], [2, 0, 1, 1]])
    assert drop_small(labels, 2).tolist() == [[2, 2, 0, 1], [0, 0, 1, 1]]


def test_perfect_probabilities_of_a_synthetic_scene_give_back_its_houses():
    """§6.3 (c) on synthetic scenes with touching rows: one instance per house, IoU ≥ 0.99. Not pixel for pixel: on the
    flat contact band of a rotated shared wall, the watershed splits by distance to the markers, which can give a pixel
    or two of the raster staircase to the neighbour."""
    for seed in range(5):
        scene = make_scene(seed)
        truth = scene.instance_map.astype(np.int32)
        inst = three_class_instances(*one_hot(class_map(scene.instance_map, 2)), 0.4, 8, 10).instance_map
        assert inst.max() == truth.max(), seed
        matched = set()
        for house in range(1, truth.max() + 1):
            ids, counts = np.unique(inst[truth == house], return_counts=True)
            best = ids[np.argmax(counts)]
            overlap = np.logical_and(truth == house, inst == best).sum()
            assert best > 0 and overlap / np.logical_or(truth == house, inst == best).sum() >= 0.99, (seed, house)
            matched.add(int(best))
        assert len(matched) == truth.max(), seed
