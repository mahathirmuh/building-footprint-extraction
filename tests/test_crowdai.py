"""CrowdAI annotation handling (docs/DATA.md §2.3): rasterisation rule, statistics and polygon conventions."""

import contextlib
import io

import numpy as np
from pycocotools.coco import COCO

from densepoly.data import crowdai
from tests.fixtures.synthetic import make_scene, to_coco


def square(x0, y0, size):
    return [x0, y0, x0 + size, y0, x0 + size, y0 + size, x0, y0 + size]


def ann(ann_id, rings):
    return {"id": ann_id, "segmentation": rings}


def test_rasterisation_matches_the_synthetic_fixture():
    """Two independent implementations of the same rule give identical instance maps."""
    scenes = [make_scene(s) for s in range(4)]
    coco = to_coco(scenes)
    for image_id, scene in enumerate(scenes, start=1):
        anns = [a for a in coco["annotations"] if a["image_id"] == image_id]
        result = crowdai.rasterize_instances(anns, 300, 300)
        np.testing.assert_array_equal(result.instance_map, scene.instance_map)
        assert (result.overlap_px, result.n_hidden, result.n_skipped, result.n_empty) == (0, 0, 0, 0)


def test_small_building_over_a_big_one_stays_visible():
    result = crowdai.rasterize_instances([ann(1, [square(5, 5, 4)]), ann(2, [square(0, 0, 20)])], 32, 32)
    assert (result.instance_map == 1).sum() == 16
    assert (result.instance_map == 2).sum() == 400 - 16
    assert result.overlap_px == 16 and result.n_hidden == 0


def test_identical_duplicates_hide_one_annotation():
    result = crowdai.rasterize_instances([ann(1, [square(2, 2, 6)]), ann(2, [square(2, 2, 6)])], 16, 16)
    assert result.n_hidden == 1 and result.overlap_px == 36


def test_unusable_rings_are_skipped_and_counted():
    anns = [ann(1, [[1, 1, 5, 5]]), ann(2, [[0, 0, 3, 0, 3]]), ann(3, [square(1, 1, 3)]), ann(4, [[2, 2, 2, 2, 2, 2]])]
    result = crowdai.rasterize_instances(anns, 8, 8)
    assert (result.n_skipped, result.n_empty, result.annotation_ids) == (2, 1, [3])
    assert (result.instance_map == 1).sum() == 9


def test_polygon_conventions_detect_the_closing_vertex_and_winding():
    clockwise = square(0, 0, 4)  # (0,0) -> (4,0) -> (4,4) -> (0,4) turns clockwise on screen (y points down)
    counterclockwise = [0, 0, 0, 4, 4, 4, 4, 0]
    closed = clockwise + clockwise[:2]
    conv = crowdai.polygon_conventions([ann(1, [clockwise]), ann(2, [counterclockwise]), ann(3, [closed])])
    assert conv["rings"] == 3 and conv["closing_vertex_repeated"] == 1
    assert conv["clockwise_on_screen"] == 2 and conv["counterclockwise_on_screen"] == 1
    assert conv["vertices_max"] == 4


def test_annotation_stats_and_file_check(tmp_path):
    dataset = to_coco([make_scene(0), make_scene(1)], file_names=["a.png", "b.png"])
    dataset["images"].append({"id": 3, "file_name": "c.png", "height": 300, "width": 300})
    for name in ("a.png", "x.png"):
        (tmp_path / name).write_bytes(b"")
    coco = COCO()
    coco.dataset = dataset
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    stats = crowdai.annotation_stats(coco)
    assert (stats["images"], stats["images_without_annotations"]) == (3, 1)
    assert stats["annotations"] == len(dataset["annotations"]) and stats["multi_ring_annotations"] == 0
    files = crowdai.check_files(coco, tmp_path)
    assert files["missing_examples"] == ["b.png", "c.png"] and files["extra_examples"] == ["x.png"]
