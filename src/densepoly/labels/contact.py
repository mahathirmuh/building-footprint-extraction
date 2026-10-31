"""Contact class (docs/METHOD.md §3.2): building pixels that have another building within r px.

Contact is restricted to building pixels, so the background gap between two houses stays class 0 and the union of
the building and contact classes equals the union of all buildings.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import disk


def contact_mask(instance_map: np.ndarray, radius: int) -> np.ndarray:
    """True where a building pixel sees a different building inside a Euclidean disk of `radius` px.

    One dilation and one erosion replace a loop over building pairs: a pixel of label l is in contact when the largest
    label nearby is not l, or the smallest non-zero label nearby is not l. On gray levels, dilation and erosion are the
    maximum and minimum filters; OpenCV runs them several times faster than scipy, which matters because this runs per
    tile in the training loader, and BORDER_REFLECT is the default border of the scipy filters.
    """
    footprint = disk(radius).astype(np.uint8)
    labels = instance_map.astype(np.uint16)
    big = np.iinfo(np.uint16).max
    largest = cv2.dilate(labels, footprint, borderType=cv2.BORDER_REFLECT)
    smallest = cv2.erode(np.where(labels > 0, labels, big).astype(np.uint16), footprint, borderType=cv2.BORDER_REFLECT)
    return (labels > 0) & ((largest != labels) | (smallest != labels))


def class_map(instance_map: np.ndarray, radius: int) -> np.ndarray:
    """Three-class label: 0 background, 1 building, 2 contact."""
    classes = (instance_map > 0).astype(np.uint8)
    classes[contact_mask(instance_map, radius)] = 2
    return classes
