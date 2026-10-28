"""The synthetic fixture is self-consistent: polygons are the truth; the raster and the image follow from them."""

import numpy as np
import pytest
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from scipy import ndimage as ndi
from shapely.geometry import Polygon
from skimage.morphology import disk

from tests.fixtures.synthetic import make_scene, to_coco

FOUR = ndi.generate_binary_structure(2, 1)


@pytest.fixture(scope="module", params=range(8), ids=lambda s: f"seed{s}")
def scene(request):
    return make_scene(request.param)


def coco_mask(polygon, height, width):
    """Independent pycocotools rasterisation (deliberately not the fixture's own helper)."""
    rle = mask_utils.frPyObjects([polygon.ravel().tolist()], height, width)
    return mask_utils.decode(rle)[..., 0].astype(bool)


def test_same_seed_gives_the_same_scene():
    a, b = make_scene(5), make_scene(5)
    assert np.array_equal(a.image, b.image) and np.array_equal(a.instance_map, b.instance_map)
    assert all(np.array_equal(p.polygon, q.polygon) for p, q in zip(a.buildings, b.buildings, strict=True))


def test_different_seeds_give_different_scenes():
    assert not np.array_equal(make_scene(0).instance_map, make_scene(1).instance_map)


def test_instance_map_is_the_pycocotools_raster_of_the_polygons(scene):
    """PLAN.md Phase 0 acceptance criterion; paint order = decreasing area, as for CrowdAI (docs/DATA.md §2.3)."""
    height, width = scene.instance_map.shape
    masks = {b.id: coco_mask(b.polygon, height, width) for b in scene.buildings}
    expected = np.zeros((height, width), np.uint16)
    for bid in sorted(masks, key=lambda i: (-masks[i].sum(), i)):
        expected[masks[bid]] = bid
    np.testing.assert_array_equal(scene.instance_map, expected)


def test_no_pixel_belongs_to_two_buildings(scene):
    height, width = scene.instance_map.shape
    for b in scene.buildings:
        assert np.array_equal(coco_mask(b.polygon, height, width), scene.instance_map == b.id), f"building {b.id}"


def test_polygons_are_valid_inside_the_tile_and_rasterise_to_one_region(scene):
    height, width = scene.instance_map.shape
    for b in scene.buildings:
        assert len(np.unique(b.polygon, axis=0)) == len(b.polygon) >= 3
        assert Polygon(b.polygon).is_valid
        assert b.polygon.min() >= 0 and b.polygon[:, 0].max() <= width and b.polygon[:, 1].max() <= height
        assert ndi.label(scene.instance_map == b.id, structure=FOUR)[1] == 1, f"building {b.id}"


def test_every_scene_has_all_building_groups(scene):
    assert scene.image.shape == (300, 300, 3) and scene.image.dtype == np.uint8
    assert scene.instance_map.dtype == np.uint16
    kinds = [b.kind for b in scene.buildings]
    assert kinds.count("rect") >= 2 and kinds.count("L") == 1
    assert 3 <= kinds.count("row") <= 5 and 3 <= kinds.count("gap_row") <= 4
    assert all(len(b.polygon) == 6 for b in scene.buildings if b.kind == "L")
    assert [b.id for b in scene.buildings] == list(range(1, len(scene.buildings) + 1))


def test_every_edge_follows_theta(scene):
    """theta_deg is the side direction mod 90 in image coordinates (docs/METHOD.md §3.3): every edge is ∥ or ⊥."""
    for b in scene.buildings:
        edges = np.roll(b.polygon, -1, axis=0) - b.polygon
        diff = (np.degrees(np.arctan2(edges[:, 1], edges[:, 0])) - b.theta_deg) % 90.0
        assert np.all(np.minimum(diff, 90.0 - diff) < 0.1), f"building {b.id}"
    assert all(5 <= b.theta_deg <= 85 for b in scene.buildings if b.kind == "rect")


def test_declared_neighbours_are_exactly_the_touching_buildings(scene):
    for b in scene.buildings:
        grown = ndi.binary_dilation(scene.instance_map == b.id, structure=FOUR)
        assert set(np.unique(scene.instance_map[grown]).tolist()) - {0, b.id} == set(b.neighbors), f"building {b.id}"


def test_isolated_houses_have_no_building_within_3_px(scene):
    """Isolated houses must never get contact pixels (contact radius r is 2 px on CrowdAI, 3 px on WHU)."""
    for b in scene.buildings:
        if b.kind in ("rect", "L"):
            near = ndi.binary_dilation(scene.instance_map == b.id, structure=disk(3))
            assert set(np.unique(scene.instance_map[near]).tolist()) <= {0, b.id}, f"building {b.id}"


def test_gaps_are_one_or_two_pixels_of_dark_ground(scene):
    assert scene.gap_mask.any() and not scene.instance_map[scene.gap_mask].any()
    for a, c, width in scene.gap_pairs:
        assert width in (1, 2)
        grown = ndi.binary_dilation(scene.instance_map == a, structure=FOUR, iterations=width)
        assert not (grown & (scene.instance_map == c)).any()  # still apart after `width` steps...
        grown = ndi.binary_dilation(grown, structure=FOUR)
        assert (grown & (scene.instance_map == c)).any()  # ...and touching after one more


def test_roofs_are_brighter_than_ground_and_gaps_darkest(scene):
    brightness = scene.image.max(axis=2).astype(float)  # b = max(R, G, B), docs/METHOD.md §2.1
    roofs = brightness[scene.instance_map > 0].mean()
    ground = brightness[(scene.instance_map == 0) & ~scene.gap_mask].mean()
    gaps = brightness[scene.gap_mask].mean()
    assert roofs > ground + 50 and ground > gaps + 50


def test_coco_export_rasterises_back_to_the_instance_maps():
    scenes = [make_scene(0), make_scene(1)]
    coco = COCO()
    coco.dataset = to_coco(scenes)
    coco.createIndex()
    for image_id, scene in enumerate(scenes, start=1):
        anns = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
        assert len(anns) == len(scene.buildings)
        for ann, b in zip(anns, scene.buildings, strict=True):
            assert np.array_equal(coco.annToMask(ann).astype(bool), scene.instance_map == b.id), f"building {b.id}"
