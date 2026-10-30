"""Statistics of the evaluation (docs/EVALUATION.md §4).

COCO AP over any list of images, duplicates allowed, from the evalImgs of a single COCOeval.evaluate(). evaluate()
drops duplicate imgIds (np.unique), so a bootstrap resample cannot simply be passed as params.imgIds. Instead the
cached evalImgs are re-indexed for the image list and pycocotools' own COCOeval.accumulate() and summarize() run on
them unchanged: with the identity list they give stats exactly. The evalImgs of the vendored boundary-iou-api COCOeval
have the same structure, so AP_boundary goes the same way (its own accumulate() fails on numpy ≥ 1.24).
"""

from __future__ import annotations

import contextlib
import copy
import io
from dataclasses import dataclass

import numpy as np
from pycocotools.cocoeval import COCOeval

STAT_NAMES = ("AP", "AP50", "AP75", "AP_S", "AP_M", "AP_L", "AR1", "AR10", "AR100", "AR_S", "AR_M", "AR_L")


@dataclass
class EvalCache:
    """What COCOeval.accumulate() needs from one evaluate(): evalImgs, ordered [category][area][image], and the
    parameters of that evaluation (img_ids sorted and unique, as evaluate() leaves them)."""

    eval_imgs: list
    params: object

    @property
    def img_ids(self) -> list[int]:
        return [int(i) for i in self.params.imgIds]

    def positions(self, image_ids) -> list[int]:
        """Positions of image ids in img_ids, in the order given (duplicates kept)."""
        index = {img_id: i for i, img_id in enumerate(self.img_ids)}
        return [index[int(i)] for i in image_ids]


def eval_cache(evaluator) -> EvalCache:
    """EvalCache of a COCOeval (pycocotools or vendored boundary-iou-api) after evaluate()."""
    return EvalCache(evaluator.evalImgs, copy.deepcopy(evaluator._paramsEval))


def _reindexed(cache: EvalCache, positions, area_labels=None, max_dets=None) -> COCOeval:
    """A pycocotools COCOeval holding the evalImgs of the images at positions, ready for accumulate()."""
    p = cache.params
    n_cats = len(p.catIds) if p.useCats else 1
    n_areas, n_images = len(p.areaRng), len(p.imgIds)
    areas = range(n_areas) if area_labels is None else [p.areaRngLbl.index(label) for label in area_labels]
    positions = list(positions)
    eval_imgs = [
        cache.eval_imgs[k * n_areas * n_images + a * n_images + i]
        for k in range(n_cats)
        for a in areas
        for i in positions
    ]
    params = copy.copy(p)
    params.imgIds = list(range(len(positions)))
    params.areaRng = [p.areaRng[a] for a in areas]
    params.areaRngLbl = [p.areaRngLbl[a] for a in areas]
    if max_dets is not None:
        params.maxDets = list(max_dets)
    params.iouType = "segm"  # summarize() only tells keypoints from boxes and masks
    shell = COCOeval.__new__(COCOeval)
    shell.evalImgs, shell.params, shell._paramsEval, shell.eval, shell.stats = eval_imgs, params, params, {}, []
    return shell


def coco_stats(cache: EvalCache, positions=None) -> dict[str, float]:
    """The 12 COCO summary numbers (STAT_NAMES) over the images at positions; None means all images."""
    shell = _reindexed(cache, range(len(cache.params.imgIds)) if positions is None else positions)
    with contextlib.redirect_stdout(io.StringIO()):
        shell.accumulate()
        shell.summarize()
    return dict(zip(STAT_NAMES, (float(s) for s in shell.stats), strict=True))


def coco_ap(cache: EvalCache, positions=None) -> float:
    """AP alone (IoU 0.50:0.95, all areas, 100 detections), equal to coco_stats()["AP"] but accumulating that one
    setting only, for the bootstrap."""
    shell = _reindexed(cache, range(len(cache.params.imgIds)) if positions is None else positions, ["all"], [100])
    with contextlib.redirect_stdout(io.StringIO()):
        shell.accumulate()
    precision = shell.eval["precision"]
    valid = precision[precision > -1]
    return float(np.mean(valid)) if valid.size else -1.0
