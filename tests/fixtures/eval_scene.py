"""A six-image evaluation scene for crowdai/test: dense, medium, sparse and empty tiles, with a miss, shifts, a
rotation, false positives and one polygon over two touching houses (tests of run_eval and of the bootstrap)."""

from __future__ import annotations

from densepoly import paths
from densepoly.utils.io import write_json
from tests.fixtures.coco_toy import RECT, make_gt, make_results, rotated, square

HOUSES = {
    1: [
        square(2, 2, 10),
        square(14, 2, 10),
        square(26, 2, 10),
        square(38, 2, 10),
        square(2, 20, 10),
        square(14, 20, 10),
    ],
    2: [square(5, 5, 15), square(30, 30, 20), square(5, 40, 12)],
    3: [square(20, 20, 20)],
    4: [rotated(RECT, 15)],
    5: [],
    6: [square(2, 2, 20), square(23, 2, 20)],  # 1 px apart
}
STRATA = {"kosong": [5], "jarang": [3, 4], "sedang": [2, 6], "padat": [1]}
SECONDARY = {"kosong": [5], "tidak_bersentuhan": [1, 2, 3, 4], "bersentuhan": [6]}


def shifted(points, dx, dy=0):
    return [(x + dx, y + dy) for x, y in points]


def predictions():
    """A miss and 1 px shifts on image 1, 2 px shifts and a false positive on image 2, an exact house on image 3, a
    rotation on image 4, a false positive on the empty image 5 and one polygon over both houses of image 6."""
    scenes = {
        1: [shifted(p, 1) for p in HOUSES[1][:5]],
        2: [*(shifted(p, 2) for p in HOUSES[2]), square(50, 5, 8)],
        3: HOUSES[3],
        4: [rotated(RECT, 20)],
        5: [square(10, 10, 10)],
        6: [[(2, 2), (43, 2), (43, 22), (2, 22)]],
    }
    results = make_results(scenes)
    for i, result in enumerate(results):  # distinct, unordered scores
        result["score"] = 0.3 + ((7 * i) % len(results)) / (2 * len(results))
    return results


def write_scene(tmp_path, monkeypatch):
    """(ground truth, splits folder): crowdai/test GT under a temporary DENSEPOLY_OUT and its density file."""
    monkeypatch.setenv("DENSEPOLY_OUT", str(tmp_path / "outputs"))
    gt = make_gt(HOUSES)
    write_json(paths.gt_file("crowdai", "test"), gt)
    splits = tmp_path / "splits"
    strata = {"main": {"strata": STRATA}, "secondary": {"strata": SECONDARY}}
    write_json(paths.density_file("crowdai", "test", splits), strata)
    return gt, splits
