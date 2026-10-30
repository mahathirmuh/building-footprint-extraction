"""Classical baseline B0, without deep learning (docs/METHOD.md §8).

MBI scaled to [0, 1] → threshold max(Otsu per tile, t_min) → opening with a disk → objects under min_object_px removed
→ holes filled → distance transform → h-maxima markers → watershed(−distance, markers, foreground) → polygons in mode mar
with score 1 (B0 has no probability, so its AP is a single precision-recall point).

- Otsu always splits a tile in two classes, even a tile without buildings: hence the lower bound t_min, tuned on
  val_tune_sub.
- The h-maxima markers are the regional maxima of the h-maxima transform, the reconstruction by dilation of
  distance − h under the distance (the extended maxima of Soille): one marker per dome higher than h.
  skimage.morphology.h_maxima returns another set, the pixels where distance − reconstruction ≥ h, i.e. the very top
  of each dome; along the ridge of a long building the distance wavers by a fraction of a pixel, so that set is a row
  of single pixels and one building got up to 5 markers.
- The foreground, its objects, the markers and the instances are 4-connected, as on the model side (§6.2). Holes are
  background components not reaching the tile border, with 8-connectivity, the dual of a 4-connected foreground.
- A foreground component without a marker (a narrow building whose distance to the border stays under h) becomes an
  instance of its own, like an unreached component of the models (§6.2 step 5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import disk, local_maxima, reconstruction

from densepoly.postproc.markers import drop_small, label4
from densepoly.postproc.watershed import Instances, watershed_instances

EIGHT = np.ones((3, 3), bool)


@dataclass(frozen=True)
class B0Params:
    t_min: float = 0.15
    open_radius_px: int = 2
    min_object_px: int = 30
    h_maxima: float = 2.0

    @classmethod
    def from_config(cls, b0) -> B0Params:
        """From the postproc.b0 section of an experiment config (configs/base.yaml)."""
        return cls(float(b0.t_min), int(b0.open_radius_px), int(b0.min_object_px), float(b0.h_maxima))


def threshold(mbi: np.ndarray, t_min: float) -> float:
    """max(Otsu of the tile, t_min); a constant tile has no Otsu threshold, so t_min alone applies."""
    if mbi.min() == mbi.max():
        return t_min
    return max(float(threshold_otsu(mbi)), t_min)


def foreground(mbi: np.ndarray, params: B0Params) -> np.ndarray:
    """Building pixels of B0: MBI above the threshold, opened, small objects removed, holes filled."""
    fg = ndi.binary_opening(mbi > threshold(mbi, params.t_min), structure=disk(params.open_radius_px))
    fg = drop_small(label4(fg), params.min_object_px) > 0
    return ndi.binary_fill_holes(fg, structure=EIGHT)


def h_maxima_markers(landscape: np.ndarray, h: float, mask: np.ndarray) -> np.ndarray:
    """Labelled markers inside mask: the regional maxima of the reconstruction of landscape − h under landscape."""
    domes = reconstruction(landscape - h, landscape, method="dilation")
    return label4(local_maxima(domes, connectivity=1, allow_borders=True) & mask)


def instances(mbi: np.ndarray, params: B0Params) -> Instances:
    """B0 instances of one tile from its MBI scaled to [0, 1]; every score is 1."""
    fg = foreground(mbi, params)
    distance = ndi.distance_transform_edt(fg)
    markers = h_maxima_markers(distance, params.h_maxima, fg)
    instance_map = watershed_instances(-distance, markers, fg, 1)
    return Instances(instance_map, np.ones(int(instance_map.max())))
