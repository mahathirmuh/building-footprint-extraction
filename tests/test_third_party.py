"""The verbatim third-party copies (third_party/README.md) stay unchanged, and the vendored boundary-iou-api works."""

import hashlib

import numpy as np
import pytest
from boundary_iou.utils.boundary_utils import mask_to_boundary

from densepoly import paths

EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
VERBATIM = {
    "pix2poly/whu_buildings_to_coco.py": "c9930d04abce3b9602618b8ea812a9c947250ffeca7d504d961d3c063a8d40fb",
    "pix2poly/polis.py": "ab845034a848f9fa91a9ba0e7a4de9a63d9004f28f3d5174f89d3e9c6040752c",
    "pix2poly/angle_eval.py": "1007612195147fe6019859baaffc8c7a2582f9ef0c67b8a49393e9e22b41d16d",
    "pix2poly/cIoU.py": "134f4f29af846fad135953958b9f6bdeba9853fd5efd0f795692256235289fc4",
    "boundary_iou/__init__.py": EMPTY,
    "boundary_iou/coco_instance_api/__init__.py": EMPTY,
    "boundary_iou/coco_instance_api/coco.py": "2fcbf65890836179db481312bb14a31cef808462e2d742f139a8970ee48255a2",
    "boundary_iou/coco_instance_api/cocoeval.py": "49edc7629d5e8a306756f0f5adefc03115725e88103d5bdef134a376fb26b777",
    "boundary_iou/utils/__init__.py": "6cbde9611ad3b83a8f8dcdb00d7a3e80c43a3190ee3e6535f98c2c19dabf4507",
    "boundary_iou/utils/boundary_utils.py": "a495cc15fdbded38681edef30b29c62014e81e864f0e68ca6062ce42db51fd9c",
    "licenses/boundary-iou-api-license.txt": "6da6c74bb714b3f7fe7eb6b817e42dd59ebfca15f292f54a9a7ad33a6a872863",
}


@pytest.mark.parametrize("name", sorted(VERBATIM))
def test_verbatim_copies_are_unchanged(name):
    data = (paths.REPO_ROOT / "third_party" / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == VERBATIM[name]


def test_mask_to_boundary_is_an_inner_band_of_two_percent_of_the_diagonal():
    mask = np.zeros((300, 300), np.uint8)
    mask[100:200, 50:150] = 1
    boundary = mask_to_boundary(mask, 0.02)  # round(0.02 · 424.3) = 8 px
    assert boundary.sum() == 100 * 100 - 84 * 84
    assert not boundary[mask == 0].any()


def test_mask_to_boundary_counts_the_image_border_as_boundary():
    boundary = mask_to_boundary(np.ones((300, 300), np.uint8), 0.02)
    assert boundary.sum() == 300 * 300 - 284 * 284
