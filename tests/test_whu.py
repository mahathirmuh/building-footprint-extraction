"""WHU conversion (docs/DATA.md §3.2): the Pix2Poly port is identical to the original code and keeps its documented
quirks; the 512² quadrants, the main ground truth from the official buildings and the patch index are checked on a
fake of the official release."""

import importlib.util
import os
from itertools import groupby
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from scipy import ndimage as ndi
from shapely.geometry import mapping

from densepoly import paths
from densepoly.data import whu
from densepoly.utils.io import read_json
from tests.fixtures.fake_whu import make_fake_whu


@pytest.fixture(scope="module")
def original():
    """The verbatim Pix2Poly script from third_party/ (importing it does not run its __main__ block)."""
    path = paths.REPO_ROOT / "third_party" / "pix2poly" / "whu_buildings_to_coco.py"
    spec = importlib.util.spec_from_file_location("pix2poly_whu_original", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    """The fake official release (tests/fixtures/fake_whu.py) with its official COCO grouped by tile."""
    root = tmp_path_factory.mktemp("data")
    tiles = make_fake_whu(root)
    official = {
        folder: whu.official_by_tile(read_json(root / "whu" / "official_coco_0.2" / f"{name}.json"))
        for folder, name in whu.OFFICIAL_COCO_NAMES.items()
    }
    return SimpleNamespace(tiles=tiles, raw=root / "whu" / "raw_0.2", official=official)


def crafted_mask():
    mask = np.zeros((224, 224), np.uint8)
    mask[20:50, 20:90] = 255  # two houses sharing a wall: one block in a binary label
    mask[100:130, 30:70] = 255  # isolated house
    mask[150:153, 150:153] = 255  # tiny 3 × 3 blob
    mask[160:210, 100:160] = 255
    mask[175:195, 120:140] = 0  # building with a 20 × 20 hole
    mask[60:70, 150:160] = 255
    mask[70:80, 160:170] = 255  # diagonal touch: one 8-connected component
    return mask


def random_masks(n=6):
    rng = np.random.default_rng(0)
    masks = []
    for _ in range(n):
        field = ndi.gaussian_filter(rng.random((224, 224)), 5)
        masks.append(((field > np.quantile(field, 0.75)) * 255).astype(np.uint8))
    return masks


MASKS = [crafted_mask(), *random_masks()]


@pytest.mark.parametrize("index", range(len(MASKS)))
def test_bmask_to_poly_is_identical_to_pix2poly(original, index):
    ours, theirs = whu.bmask_to_poly(MASKS[index], 1), original.bmask_to_poly(MASKS[index], 1)
    assert [p.wkb for p in ours] == [p.wkb for p in theirs]


def test_crop2patch_is_identical_and_flush_with_the_border(original):
    for shape in [(512, 512, 3), (1024, 1024, 3), (224, 224, 3), (600, 500, 3)]:
        ours, theirs = whu.crop2patch(shape, 224, 224, 34), original.crop2patch(shape, 224, 224, 34)
        assert [[int(v) for v in p] for p in ours] == [[int(v) for v in p] for p in theirs]
    assert sorted({int(p[0]) for p in whu.crop2patch((512, 512, 3), 224, 224, 34)}) == [0, 190, 288]


def pix2poly_patch_loop(original, image_data, gt_im_data, label_name, im_id=0, ob_id=0):
    """Lines 308-369 of the original __main__ block, transcribed: file writes removed and the loop over the empty
    rotation list collapsed into the single unrotated patch it yields."""
    images, annotations = [], []
    patch_width = patch_height = patch_size = 224
    patch_overlap = 34
    for x_ul, y_ul, _pw, _ph in original.crop2patch(image_data.shape, patch_width, patch_height, patch_overlap):
        p_gt = gt_im_data[y_ul : y_ul + patch_height, x_ul : x_ul + patch_width]
        p_im = image_data[y_ul : y_ul + patch_height, x_ul : x_ul + patch_width, :]
        p_im, p_gt = original.lt_crop(p_im, p_gt, patch_size)
        if np.sum(p_gt > 0) > 5:
            for poly in original.bmask_to_poly(p_gt, 1):
                p_area = round(poly.area, 2)
                if p_area > 0:
                    b = poly.bounds
                    p_bbox = [b[0], b[1], b[2] - b[0], b[3] - b[1]]
                    if p_bbox[2] > 5 and p_bbox[3] > 5:
                        p_seg = [np.asarray(part).ravel().tolist() for part in mapping(poly)["coordinates"]]
                        annotations.append(
                            {
                                "id": ob_id,
                                "image_id": im_id,
                                "segmentation": p_seg,
                                "area": p_area,
                                "bbox": p_bbox,
                                "category_id": 100,
                                "iscrowd": 0,
                            }
                        )
                        ob_id += 1
        else:
            annotations.append(
                {
                    "id": ob_id,
                    "image_id": im_id,
                    "segmentation": [],
                    "area": 0.0,
                    "bbox": [],
                    "category_id": 100,
                    "iscrowd": 0,
                }
            )
            ob_id += 1
        images.append({"id": im_id, "file_name": f"{label_name}-{im_id}.tif", "width": 224, "height": 224})
        im_id += 1
    return images, annotations


def test_p2p_ground_truth_is_pix2poly_run_on_every_kept_quadrant(original, fake):
    """Each 512² quadrant with buildings is one Pix2Poly tile; ids run on across quadrants and tiles."""
    result = whu.convert_split(fake.raw / "train", "train")
    images, annotations = [], []
    for stem, tile in sorted(fake.tiles["train"].items()):
        for q, y0, x0 in whu.quadrants(tile.label):
            image, label = (a[y0 : y0 + whu.QUADRANT, x0 : x0 + whu.QUADRANT] for a in (tile.image, tile.label))
            more = pix2poly_patch_loop(original, image, label, f"{stem}_{q}", len(images), len(annotations))
            images += more[0]
            annotations += more[1]
    assert result.images == images and result.p2p == annotations and result.main is None
    assert result.counts["quadrants_kept"] == 7 and result.counts["patches"] == 63  # quadrant 3 of 10001 is empty
    assert "10002" not in {p["tile"] for p in result.patches}  # all-zero int8 label: no quadrant kept


def test_main_ground_truth_keeps_touching_houses_apart():
    p_inst = np.zeros((224, 224), np.uint16)
    p_inst[40:80, 30:60] = 1
    p_inst[40:80, 60:95] = 2  # shares a wall with building 1: one blob for Pix2Poly
    p_inst[150:180, 150:190] = 3
    p2p = whu.patch_annotations((p_inst > 0).astype(np.uint8) * 255, 0, 0)
    main = whu.instance_patch_annotations(p_inst, 0, 0)
    assert len(p2p) == 2 and len(main) == 3 and [a["id"] for a in main] == [0, 1, 2]
    assert main[2]["segmentation"] == p2p[1]["segmentation"]  # an isolated building is traced the same way
    assert main[0]["bbox"] == [30.0, 40.0, 29.0, 39.0]  # half a pixel inside its own mask, as in Pix2Poly


def test_main_negative_sample_counts_all_building_pixels():
    p_inst = np.zeros((224, 224), np.uint16)
    p_inst[5, 5:8] = 1
    p_inst[9, 5:7] = 2  # 5 building pixels in two buildings
    expected = {"id": 9, "image_id": 4, "segmentation": [], "area": 0.0, "bbox": [], "category_id": 100, "iscrowd": 0}
    assert whu.instance_patch_annotations(p_inst, image_id=4, first_ann_id=9) == [expected]


def test_quadrants_keep_only_quadrants_with_buildings():
    label = np.zeros((1024, 1024), np.uint8)
    label[10, 600] = 255
    label[700, 700] = 255
    assert whu.quadrants(label) == [(1, 0, 512), (3, 512, 512)]
    with pytest.raises(ValueError, match="quadrants"):
        whu.quadrants(np.zeros((600, 1024), np.uint8))


def test_convert_split_main_ground_truth_and_patch_offsets(fake):
    result = whu.convert_split(fake.raw / "train", "train", fake.official["train"])
    assert [img["id"] for img in result.images] == list(range(63))
    assert [a["id"] for a in result.main] == list(range(len(result.main)))
    assert result.counts["main"]["building_annotations"] > result.counts["p2p"]["building_annotations"]  # rows
    assert result.counts["official_in_used_tiles"]["annotations"] == sum(
        len(t.buildings) for t in fake.tiles["train"].values()
    )
    for patch in result.patches:  # the offset gives back the patch: Pix2Poly's annotations of that crop
        label = fake.tiles["train"][patch["tile"]].label
        p_gt = label[patch["y"] : patch["y"] + 224, patch["x"] : patch["x"] + 224]
        got = [a["segmentation"] for a in result.p2p if a["image_id"] == patch["image_id"]]
        assert got == [a["segmentation"] for a in whu.patch_annotations(p_gt, 0, 0)]
        assert result.images[patch["image_id"]]["file_name"] == patch["file_name"]


def test_test_split_keeps_part_300_only(fake):
    result = whu.convert_split(fake.raw / "test", "test", fake.official["test"])
    assert (result.counts["tiles"], result.counts["tiles_used"], result.counts["patches"]) == (2, 1, 27)
    assert {p["tile"] for p in result.patches} == {"3000"}


def test_parallel_conversion_matches_serial(fake):
    serial = whu.convert_split(fake.raw / "val", "val", fake.official["val"], workers=1)
    parallel = whu.convert_split(fake.raw / "val", "val", fake.official["val"], workers=2)
    assert (parallel.images, parallel.main, parallel.p2p, parallel.patches) == (
        serial.images,
        serial.main,
        serial.p2p,
        serial.patches,
    )


def test_negative_label_values_are_rejected(tmp_path):
    for kind, array in (("image", np.zeros((1024, 1024, 3), np.uint8)), ("label", np.full((1024, 1024), -1, np.int8))):
        (tmp_path / kind).mkdir()
        tifffile.imwrite(tmp_path / kind / "1.tif", array, photometric="rgb" if kind == "image" else "minisblack")
    with pytest.raises(ValueError, match="negative"):
        whu.convert_tile(tmp_path, "1.tif", None)


def test_official_by_tile_lists_every_tile(fake):
    assert set(fake.official["train"]) == {"10000", "10001", "10002"} and fake.official["train"]["10002"] == []


def test_negative_patch_gets_one_empty_annotation():
    p_gt = np.zeros((224, 224), np.uint8)
    p_gt[5, 5:10] = 255  # 5 building pixels still count as a negative patch
    expected = {"id": 3, "image_id": 7, "segmentation": [], "area": 0.0, "bbox": [], "category_id": 100, "iscrowd": 0}
    assert whu.patch_annotations(p_gt, image_id=7, first_ann_id=3) == [expected]


def test_patch_with_only_tiny_buildings_gets_no_annotation_at_all():
    p_gt = np.zeros((224, 224), np.uint8)
    p_gt[10:13, 10:13] = 255  # 9 pixels (> 5), but the traced polygon is only 2 × 2 px
    assert whu.patch_annotations(p_gt, 0, 0) == []


def test_polygon_runs_through_boundary_pixel_centres():
    p_gt = np.zeros((224, 224), np.uint8)
    p_gt[50:80, 60:100] = 255  # a 30 × 40 px building
    (ann,) = whu.patch_annotations(p_gt, 0, 0)
    ring = np.asarray(ann["segmentation"][0]).reshape(-1, 2)
    assert np.array_equal(ring[0], ring[-1])  # the closing vertex is repeated
    assert ann["area"] == 39 * 29 and ann["bbox"] == [60.0, 50.0, 39.0, 29.0]  # half a pixel inside the mask


def test_holes_larger_than_10_px_become_extra_rings():
    p_gt = np.zeros((224, 224), np.uint8)
    p_gt[40:100, 40:100] = 255
    p_gt[60:80, 60:80] = 0
    (ann,) = whu.patch_annotations(p_gt, 0, 0)
    assert len(ann["segmentation"]) == 2


def pix2poly_tile_name(label_file, split):
    """Lines 298-302 of the original script: the first two digit/non-digit runs of the name, four for test."""
    label_info = ["".join(g) for _, g in groupby(label_file, key=lambda x: x.isdigit())]
    return "".join(label_info[:4]) if split == "test" else "".join(label_info[:2])


def test_file_stem_is_the_name_pix2poly_parses_wherever_its_parser_works():
    for name, split in (("tile_123.tif", "train"), ("tile_12_34.tif", "test")):
        assert pix2poly_tile_name(name, split) == os.path.splitext(name)[0]
    # the official names are digits only: Pix2Poly would look for 10000.tif.tif
    assert pix2poly_tile_name("10000.tif", "train") == "10000.tif"


def test_val_split_keeps_official_tiles_whole():
    names = [f"{20000 + t}_{q}-{(4 * t + q) * 9 + k}.tif" for t in range(20) for q in range(4) for k in range(9)]
    sub, select = whu.split_val_by_tile(names, tune_sub_size=40, seed=0)

    def tiles(items):
        return {n.split("_")[0] for n in items}

    assert not tiles(sub) & tiles(select) and sorted(sub + select) == sorted(names)
    assert 40 <= len(sub) < 40 + 36 and len(sub) % 36 == 0  # 36 patches per official tile
    assert sub == [n for n in names if n in set(sub)]  # patch order is kept


def test_touching_pairs():
    def square(x, y):
        return [x, y, x + 10, y, x + 10, y + 10, x, y + 10]

    anns = [
        {"image_id": 1, "segmentation": [square(0, 0)]},
        {"image_id": 1, "segmentation": [square(10, 0)]},  # shares a wall with the first
        {"image_id": 1, "segmentation": [square(30, 0)]},  # 10 px away
        {"image_id": 2, "segmentation": [square(0, 0)]},
        {"image_id": 2, "segmentation": [square(5, 5)]},  # overlaps
        {"image_id": 3, "segmentation": []},  # negative sample
    ]
    assert whu.touching_pairs(anns) == 2
