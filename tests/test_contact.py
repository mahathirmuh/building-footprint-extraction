"""Contact labels (docs/METHOD.md §3.2) on cases with a known answer."""

import numpy as np
import pytest
from scipy import ndimage as ndi

from densepoly.labels.contact import class_map, contact_mask
from tests.fixtures.synthetic import make_scene


def two_houses_and_an_isolated_one():
    labels = np.zeros((64, 64), np.uint16)
    labels[10:30, 10:20] = 1
    labels[10:30, 20:30] = 2  # shares the wall x = 20 with house 1
    labels[40:50, 40:50] = 3  # isolated
    return labels


@pytest.mark.parametrize("radius", [1, 2, 3])
def test_contact_is_a_band_of_width_r_on_each_side_of_the_shared_wall(radius):
    contact = contact_mask(two_houses_and_an_isolated_one(), radius)
    assert np.flatnonzero(contact.any(axis=0)).tolist() == list(range(20 - radius, 20 + radius))
    assert contact[10:30, 20 - radius : 20 + radius].all()
    assert not contact[40:50, 40:50].any()


def test_class_map_keeps_the_union_of_buildings():
    labels = two_houses_and_an_isolated_one()
    classes = class_map(labels, 2)
    assert set(np.unique(classes).tolist()) == {0, 1, 2}
    assert np.array_equal(classes > 0, labels > 0)  # contact is carved out of buildings, never out of background


@pytest.fixture(scope="module", params=range(4), ids=lambda s: f"seed{s}")
def scene(request):
    return make_scene(request.param)


def expected_contact_ids(scene, radius):
    """Touching houses, plus gap-row houses whose gap leaves their pixel centres within the radius."""
    ids = {b.id for b in scene.buildings if b.neighbors}
    for a, c, width in scene.gap_pairs:
        if width + 1 <= radius:
            ids |= {a, c}
    return ids


@pytest.mark.parametrize("radius", [1, 2, 3])
def test_only_touching_or_nearly_touching_houses_get_contact(scene, radius):
    contact = contact_mask(scene.instance_map, radius)
    found = set(np.unique(scene.instance_map[contact]).tolist())
    assert found == expected_contact_ids(scene, radius)


def test_contact_matches_a_distance_transform_definition(scene):
    """Independent check: a building pixel is contact iff another building lies within r (Euclidean)."""
    radius = 2
    contact = contact_mask(scene.instance_map, radius)
    for b in scene.buildings:
        own = scene.instance_map == b.id
        others = (scene.instance_map > 0) & ~own
        near_other = ndi.distance_transform_edt(~others) <= radius
        np.testing.assert_array_equal(contact & own, own & near_other, err_msg=f"building {b.id}")
