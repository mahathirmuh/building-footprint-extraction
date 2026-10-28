"""Contact class (docs/METHOD.md §3.2): building pixels that have another building within r px.

Contact is restricted to building pixels, so the background gap between two houses stays class 0 and the union of
the building and contact classes equals the union of all buildings.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import disk


def contact_mask(instance_map: np.ndarray, radius: int) -> np.ndarray:
    """True where a building pixel sees a different building inside a Euclidean disk of `radius` px.

    One maximum and one minimum filter replace a loop over building pairs: a pixel of label l is in contact when the
    largest label nearby is not l, or the smallest non-zero label nearby is not l.
    """
    footprint = disk(radius)
    big = np.iinfo(instance_map.dtype).max
    largest = ndi.maximum_filter(instance_map, footprint=footprint)
    smallest = ndi.minimum_filter(np.where(instance_map > 0, instance_map, big), footprint=footprint)
    return (instance_map > 0) & ((largest != instance_map) | (smallest != instance_map))


def class_map(instance_map: np.ndarray, radius: int) -> np.ndarray:
    """Three-class label: 0 background, 1 building, 2 contact."""
    classes = (instance_map > 0).astype(np.uint8)
    classes[contact_mask(instance_map, radius)] = 2
    return classes
