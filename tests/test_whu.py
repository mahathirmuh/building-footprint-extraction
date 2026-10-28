"""WHU protocol A port (docs/DATA.md §3.2): identical to Pix2Poly's original code, plus its documented quirks."""

import importlib.util

import numpy as np
import pytest
from scipy import ndimage as ndi
from shapely.geometry import mapping

from densepoly import paths
from densepoly.data import whu
from densepoly.utils.io import read_json
from tests.fixtures.fake_whu import make_tile


@pytest.fixture(scope="module")
def original():
    """The verbatim Pix2Poly script from third_party/ (importing it does not run its __main__ block)."""
    path = paths.REPO_ROOT / "third_party" / "pix2poly" / "whu_buildings_to_coco.py"
    spec = importlib.util.spec_from_file_location("pix2poly_whu_original", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_convert_split_reproduces_the_pix2poly_patch_loop(original, tmp_path):
    from skimage import io

    image, label = make_tile(7)
    for kind, array in (("image", image), ("label", label)):
        (tmp_path / "raw" / kind).mkdir(parents=True)
        io.imsave(str(tmp_path / "raw" / kind / "tile_5_1.tif"), array, check_contrast=False)
    counts, names = whu.convert_split(tmp_path / "raw", tmp_path / "out", "test")
    ours = read_json(tmp_path / "out" / "annotation.json")
    images, annotations = pix2poly_patch_loop(original, image, label, "tile_5_1")
    assert ours["images"] == images and ours["annotations"] == annotations
    assert counts["patches"] == len(names) == 9 and names[0] == "tile_5_1-0.tif"
    assert len(list((tmp_path / "out" / "images").glob("*.tif"))) == 9


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


def test_tile_names_follow_pix2poly_parsing():
    assert whu.tile_name("tile_123.tif", "train") == "tile_123"
    assert whu.tile_name("tile_12_34.tif", "test") == "tile_12_34"


def test_val_split_keeps_tiles_whole():
    names = [f"tile{t}-{t * 9 + k}.tif" for t in range(20) for k in range(9)]
    sub, select = whu.split_val_by_tile(names, tune_sub_size=40, seed=0)

    def tiles(items):
        return {n.rsplit("-", 1)[0] for n in items}

    assert not tiles(sub) & tiles(select) and sorted(sub + select) == sorted(names)
    assert 40 <= len(sub) < 40 + 9 and len(sub) % 9 == 0
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
