"""Oracle post-processing (docs/EVALUATION.md §6.3): the post-processing applied to the ground truth itself, which
bounds what each stage can reach.

(a) Round trip: GT polygon → pycocotools raster → marching-squares ring → raster, IoU per annotation.
(b) The GT instance map, i.e. the training label (docs/METHOD.md §3.1, rebuilt with the function that made the cache)
    → polygonize in mode raw, mar or theta. Mode theta gets the θ of the GT polygon, the θ_GT of the θ error.
(c) The GT three-class map as perfect probabilities → markers → watershed → polygonize raw, which tests the instance
    stage alone.
Every prediction gets score 1.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from densepoly.data.crowdai import RasterResult, annotation_mask, rasterize_instances
from densepoly.labels.contact import class_map
from densepoly.labels.orientation import points_theta
from densepoly.postproc.contours import mask_to_exterior
from densepoly.postproc.polygonize import PolygonParams, polygonize, to_results
from densepoly.postproc.watershed import three_class_instances


def _annotations_by_image(gt: dict) -> dict[int, list[dict]]:
    by_image = {img["id"]: [] for img in gt["images"]}
    for ann in gt["annotations"]:
        by_image[ann["image_id"]].append(ann)
    return by_image


def round_trip_ious(gt: dict) -> np.ndarray:
    """(a) IoU between the raster of every annotation with a non-empty mask and the raster of its marching-squares
    ring. A mask in several 4-connected parts keeps only its largest ring, as an instance would."""
    sizes = {img["id"]: (img["height"], img["width"]) for img in gt["images"]}
    ious = []
    for ann in gt["annotations"]:
        height, width = sizes[ann["image_id"]]
        mask = annotation_mask(ann["segmentation"], height, width)
        if mask is None or not mask.any():
            continue
        again = annotation_mask([mask_to_exterior(mask).ravel().tolist()], height, width)
        ious.append((mask & again).sum() / (mask | again).sum())
    return np.asarray(ious, float)


def instance_maps(gt: dict) -> Iterator[tuple[dict, RasterResult]]:
    """(image, rasterised instances) of every image: the training label of docs/METHOD.md §3.1."""
    by_image = _annotations_by_image(gt)
    for image in gt["images"]:
        yield image, rasterize_instances(by_image[image["id"]], image["height"], image["width"])


def gt_theta(raster: RasterResult, annotations: dict[int, dict]) -> np.ndarray:
    """θ (radians) of the GT polygon (first ring) behind every instance of a raster."""
    return np.array(
        [points_theta(np.reshape(annotations[i]["segmentation"][0], (-1, 2))) for i in raster.annotation_ids]
    )


def instance_predictions(gt: dict, params: PolygonParams) -> list[dict]:
    """(b) Predictions of polygonize on the GT instance maps; mode theta uses the θ of the GT polygons."""
    category = gt["categories"][0]["id"]
    annotations = {ann["id"]: ann for ann in gt["annotations"]}
    results = []
    for image, raster in instance_maps(gt):
        instance_map = raster.instance_map.astype(np.int32)
        theta = gt_theta(raster, annotations) if params.mode == "theta" else None
        polygons = polygonize(instance_map, params, theta=theta)
        results += to_results(polygons, np.ones(len(raster.annotation_ids)), image["id"], category)
    return results


def watershed_predictions(
    gt: dict, params: PolygonParams, contact_radius: int, tau: float, marker_min_px: int, min_area: int
) -> list[dict]:
    """(c) Predictions of the three-class instance stage on perfect probabilities of the GT label."""
    category = gt["categories"][0]["id"]
    results = []
    for image, raster in instance_maps(gt):
        classes = class_map(raster.instance_map, contact_radius)
        p_bg, p_b, p_k = ((classes == c).astype(np.float32) for c in (0, 1, 2))
        instances = three_class_instances(p_bg, p_b, p_k, tau, marker_min_px, min_area)
        results += to_results(polygonize(instances.instance_map, params), instances.scores, image["id"], category)
    return results
