"""Vendored Pix2Poly metrics (third_party/pix2poly_metrics) against the verbatim copies (third_party/pix2poly).

docs/EVALUATION.md §2.3: on every case the wrapper changes do not touch, both versions give identical numbers; the
touched cases (mask_gti reset, PoLiS without a valid image, MTA of a cosine rounded above 1) behave as documented.
Also the synthetic checks of §6.1 that concern the vendored functions themselves (PoLiS of a shifted square, MTA of a
rotated polygon).
"""

import importlib.util
import json
import re
import sys
import types

import numpy as np
import pytest
from pix2poly_metrics import angle_eval, cIoU, polis
from pycocotools import mask as cocomask
from pycocotools.coco import COCO
from shapely import affinity
from shapely.geometry import Polygon

from densepoly import paths

SIZE = 64
CATEGORY = 100
RECT = [(8, 20), (48, 20), (48, 36), (8, 36)]
NO_GT_IMAGE = 6


def square(x, y, side):
    return [(x, y), (x + side, y), (x + side, y + side), (x, y + side)]


def rotated(points, degrees):
    return list(affinity.rotate(Polygon(points), degrees, origin="centroid").exterior.coords)[:-1]


def ring(points):
    """COCO ring that repeats its first point at the end (CrowdAI convention)."""
    return [float(v) for point in [*points, points[0]] for v in point]


# {image_id: [segmentation, ...]}; a segmentation is a list of rings, [] is a WHU negative sample.
GT = {
    1: [[ring(square(10, 10, 20))]],
    2: [[ring(RECT)]],
    3: [[ring(square(5, 5, 12))], [ring(square(35, 35, 15)), ring(square(39, 39, 5))]],  # second one has a hole
    4: [[ring(square(20, 20, 16))]],
    8: [[]],
}
DT = {
    1: [ring(square(12, 10, 20))],  # shifted 2 px along x
    2: [ring(rotated(RECT, 10))],
    3: [ring(square(5, 5, 12)), ring(square(45, 5, 8))],  # one exact match, one false positive
    8: [ring(square(30, 30, 10))],  # false positive on a negative sample
}


def write_gt(path, image_ids):
    annotations = []
    for image_id, segmentations in GT.items():
        for segmentation in segmentations:
            if segmentation:
                x0, y0, x1, y1 = Polygon(np.reshape(segmentation[0], (-1, 2))).bounds
                area, bbox = Polygon(np.reshape(segmentation[0], (-1, 2))).area, [x0, y0, x1 - x0, y1 - y0]
            else:
                area, bbox = 0.0, []
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": CATEGORY,
                    "iscrowd": 0,
                    "segmentation": segmentation,
                    "area": area,
                    "bbox": bbox,
                }
            )
    images = [{"id": i, "width": SIZE, "height": SIZE, "file_name": f"{i}.png"} for i in image_ids]
    path.write_text(json.dumps({"images": images, "categories": [{"id": CATEGORY}], "annotations": annotations}))


def bbox(segmentation_ring):
    x0, y0, x1, y1 = Polygon(np.reshape(segmentation_ring, (-1, 2))).bounds
    return [x0, y0, x1 - x0, y1 - y0]


def write_results(path, detections):
    """COCO results with a bbox: without one, pycocotools loadRes reads polygon segmentations as RLE and fails."""
    results = [
        {"image_id": i, "category_id": CATEGORY, "segmentation": [r], "score": 1.0, "bbox": bbox(r)}
        for i, rings in detections.items()
        for r in rings
    ]
    path.write_text(json.dumps(results))


@pytest.fixture
def scene(tmp_path):
    """(gt_path, dt_path); with_no_gt_image adds an image that has predictions but no GT annotation, iterated right
    after image 3, at the place of image 3's first building; drop removes the predictions of some images."""

    def make(with_no_gt_image=False, drop=()):
        image_ids = sorted([*GT, *([NO_GT_IMAGE] if with_no_gt_image else [])])
        detections = {**DT, **({NO_GT_IMAGE: [ring(square(5, 5, 12))]} if with_no_gt_image else {})}
        detections = {i: rings for i, rings in detections.items() if i not in drop}
        name = f"{with_no_gt_image}_{'-'.join(map(str, drop))}"
        gt_path, dt_path = tmp_path / f"gt_{name}.json", tmp_path / f"dt_{name}.json"
        write_gt(gt_path, image_ids)
        write_results(dt_path, detections)
        return str(gt_path), str(dt_path)

    return make


def load(gt_path, dt_path):
    """Fresh COCO objects for every metric: polis.py rewrites empty GT segmentations in place."""
    gt = COCO(gt_path)
    return gt, gt.loadRes(dt_path)


def original(name, monkeypatch):
    """The verbatim Pix2Poly module; descartes (imported by angle_eval for debug plots only) is stubbed."""
    descartes = types.ModuleType("descartes")
    descartes.PolygonPatch = None
    monkeypatch.setitem(sys.modules, "descartes", descartes)
    spec = importlib.util.spec_from_file_location(
        f"pix2poly_original_{name}", paths.REPO_ROOT / "third_party" / "pix2poly" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def printed(text, label):
    return float(re.search(rf"{re.escape(label)}:\s+(\S+)", text).group(1))


def test_ciou_equals_the_original_where_the_wrapper_changes_nothing(scene, monkeypatch, capsys):
    gt, dt = scene()
    original("cIoU", monkeypatch).compute_IoU_cIoU(dt, gt)
    out = capsys.readouterr().out
    result = cIoU.compute_IoU_cIoU(dt, gt, progress=False)
    assert result["iou"] == printed(out, "Mean IoU")
    assert result["ciou"] == printed(out, "Mean C-IoU")
    assert result["n_rel_diff"] == printed(out, "Mean N-Relative Difference")
    assert result["n_ratio"] == printed(out, "Mean N-Ratio")
    assert sorted(result["per_image"]) == [1, 2, 3, 8]  # only images with predictions (Pix2Poly rule)
    assert result["per_image"][8] == {"iou": 0.0, "ciou": 0.0, "n": 5, "n_gt": 0}  # negative sample: empty GT


def test_ciou_reset_changes_only_the_iou_of_an_image_without_gt(scene, monkeypatch, capsys):
    gt, dt = scene(with_no_gt_image=True)
    original("cIoU", monkeypatch).compute_IoU_cIoU(dt, gt)
    out = capsys.readouterr().out
    result = cIoU.compute_IoU_cIoU(dt, gt, progress=False)
    assert result["per_image"][NO_GT_IMAGE] == {"iou": 0.0, "ciou": 0.0, "n": 5, "n_gt": 0}

    # The original compares that image with the GT mask left over from image 3 ...
    coco = COCO(gt)
    stale_gt = np.any([coco.annToMask(a) for a in coco.loadAnns(coco.getAnnIds(imgIds=3))], axis=0)
    prediction = cocomask.decode(cocomask.frPyObjects([ring(square(5, 5, 12))], SIZE, SIZE))[..., 0]
    stale_iou = cIoU.calc_IoU(prediction, stale_gt)
    n = result["n_images"]
    assert stale_iou > 0.3
    assert printed(out, "Mean IoU") * n == pytest.approx(result["iou"] * n + stale_iou)
    # ... but its C-IoU is ~0 either way, because N_GT is already reset there (ps = 1e-9 / N).
    assert printed(out, "Mean C-IoU") == pytest.approx(result["ciou"], abs=1e-9)


def test_polis_equals_the_original(scene, monkeypatch):
    gt, dt = scene()
    expected = original("polis", monkeypatch).PolisEval(*load(gt, dt)).evaluate()
    wrapped = polis.PolisEval(*load(gt, dt))
    assert wrapped.evaluate() == expected
    # Image 3 matches exactly (value 0) and images 4 and 8 have no pair, so the pooling skips them.
    assert wrapped.num_valid_imgs == 2
    assert wrapped.per_image[1] == pytest.approx(0.4 * 2)
    assert {i for i, v in wrapped.per_image.items() if v == 0} == {3, 4, 8}


def test_polis_returns_none_when_no_image_is_valid(scene, monkeypatch, tmp_path):
    gt, _ = scene()
    gt_as_prediction = tmp_path / "gt_as_dt.json"
    write_results(gt_as_prediction, {i: [s[0] for s in segs if s] for i, segs in GT.items()})
    with pytest.raises(ZeroDivisionError):
        original("polis", monkeypatch).PolisEval(*load(gt, str(gt_as_prediction))).evaluate()
    wrapped = polis.PolisEval(*load(gt, str(gt_as_prediction)))
    assert wrapped.evaluate() is None
    assert wrapped.num_valid_imgs == 0


def test_mta_equals_the_original(scene, monkeypatch):
    # Without the prediction on the negative sample (image 8): the original raises there (next test).
    gt, dt = scene(drop=(8,))
    expected = original("angle_eval", monkeypatch).ContourEval(*load(gt, dt)).evaluate()
    wrapped = angle_eval.ContourEval(*load(gt, dt))
    assert len(expected) > 0
    assert np.array_equal(wrapped.evaluate(), expected)
    per_image = wrapped.evaluate_per_image()
    assert np.array_equal(np.concatenate([per_image[i] for i in wrapped.img_ids]), expected)
    assert len(per_image[4]) == 0  # no prediction


def test_mta_skips_an_image_without_gt_polygons(scene):
    """Wrapper change 4: a prediction on an image without GT polygons adds no value (its precision is 0), as the
    precision filter intends; shapely 2.2 / GEOS 3.14 would raise inside that filter instead."""
    with_prediction = angle_eval.ContourEval(*load(*scene())).evaluate_per_image()
    without = angle_eval.ContourEval(*load(*scene(drop=(8,)))).evaluate_per_image()
    assert len(with_prediction[8]) == 0
    assert all(np.array_equal(with_prediction[i], without[i]) for i in without)


def test_mta_of_exactly_parallel_edges_is_zero_where_the_original_gives_nan(monkeypatch):
    """Wrapper change 5: small diamonds inside a 45° diamond keep only edges parallel to the GT contour (the others
    project onto one point); their cosine can round to 1 + 2.2e-16, where the original returns NaN instead of 0."""
    verbatim = original("angle_eval", monkeypatch)
    gt = {"segmentation": [ring([(50, 10), (90, 50), (50, 90), (10, 50)])]}
    pairs = []
    for cx in np.arange(30, 40, 0.5):
        dt = {"segmentation": [ring([(cx, 48.5), (cx + 1.5, 50), (cx, 51.5), (cx - 1.5, 50)])]}
        with np.errstate(invalid="ignore"):
            (expected,) = verbatim.compute_contour_metrics(([gt], [dt]))
        (wrapped,) = angle_eval.compute_contour_metrics(([gt], [dt]))
        pairs.append((expected, wrapped))
    assert any(np.isnan(expected) for expected, _ in pairs)
    assert all(wrapped == (0.0 if np.isnan(expected) else expected) for expected, wrapped in pairs)


def test_subsets_keep_the_values_of_their_images(scene):
    gt, dt = scene()
    full = cIoU.compute_IoU_cIoU(dt, gt, progress=False)
    part = cIoU.compute_IoU_cIoU(dt, gt, img_ids=[2, 8], progress=False)
    assert part["per_image"] == {i: full["per_image"][i] for i in (2, 8)}

    polis_full = polis.PolisEval(*load(gt, dt))
    polis_full.evaluate()
    polis_part = polis.PolisEval(*load(gt, dt), imgIds=[1, 2])
    assert polis_part.evaluate() == pytest.approx((polis_full.per_image[1] + polis_full.per_image[2]) / 2)
    assert polis_part.per_image == {i: polis_full.per_image[i] for i in (1, 2)}

    mta_full = angle_eval.ContourEval(*load(gt, dt)).evaluate_per_image()
    mta_part = angle_eval.ContourEval(*load(gt, dt), img_ids=[1, 2]).evaluate_per_image()
    assert sorted(mta_part) == [1, 2]
    assert all(np.array_equal(mta_part[i], mta_full[i]) for i in (1, 2))


@pytest.mark.parametrize("d", [0.5, 2.0, 3.5])
def test_polis_of_a_shifted_square_is_0_4_d(d):
    """§6.1: two vertices per square at distance d, 4d in total, divided by 2·5 (the closing point is counted)."""
    a = Polygon(square(10, 10, 20))
    assert polis.compare_polys(a, affinity.translate(a, xoff=d)) == pytest.approx(0.4 * d)


@pytest.mark.parametrize("alpha", [5, 10, 20])
def test_mta_of_a_rotated_polygon_is_at_least_alpha(alpha):
    """§6.1, per pair (one GT, one prediction): rotating by alpha about the centre gives MTA ≥ alpha − 0.5°."""
    gt, dt = {"segmentation": [ring(RECT)]}, {"segmentation": [ring(rotated(RECT, alpha))]}
    values = angle_eval.compute_contour_metrics(([gt], [dt]))
    assert len(values) == 1
    assert np.degrees(values[0]) >= alpha - 0.5


def test_mta_of_identical_polygons_is_almost_zero():
    """Not exactly 0: the GT is buffered by 1e-4, which rounds its corners."""
    gt = {"segmentation": [ring(RECT)]}
    values = angle_eval.compute_contour_metrics(([gt], [gt]))
    assert len(values) == 1
    assert 0 <= np.degrees(values[0]) < 0.01
