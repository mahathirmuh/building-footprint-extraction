"""Training-time transforms (docs/METHOD.md §2.4, §3.5).

1. Random dihedral geometry (flip H, flip V, rot90 by k, transpose) on RGB, priors and the instance map together; on
   arrays these are exact index permutations, so the instance map stays nearest-neighbour.
2. Targets (classes, θ, V, w) derived from the transformed instance map, so they always match the tile; transforming
   the targets instead would need the sign rules of sin 4θ (sin4_sign), which the tests use as the reference.
3. Light brightness/contrast change on RGB only, since the priors were computed from the original image.
4. Padding at the bottom and right to pad_to: image and priors reflect-padded (numpy "reflect", i.e. OpenCV's
   REFLECT_101), classes ignore_index, θ targets 0, V false, w 0. Predictions are cropped back to the tile size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from densepoly.labels.targets import Targets, targets_from_config


def random_geometry(rng: np.random.Generator, augment) -> list[tuple[str, int]]:
    """Ops drawn independently with the probabilities of the augment config: flip_h, flip_v, rot90 (k in 1-3) and
    transpose, applied in that order."""
    ops = []
    if rng.random() < augment.p_flip_h:
        ops.append(("flip_h", 0))
    if rng.random() < augment.p_flip_v:
        ops.append(("flip_v", 0))
    if rng.random() < augment.p_rot90:
        ops.append(("rot90", int(rng.integers(1, 4))))
    if rng.random() < augment.p_transpose:
        ops.append(("transpose", 0))
    return ops


def apply_geometry(array: np.ndarray, ops: list[tuple[str, int]]) -> np.ndarray:
    """Apply ops to the first two axes of an (H, W) or (H, W, C) array."""
    for name, k in ops:
        if name == "flip_h":
            array = array[:, ::-1]
        elif name == "flip_v":
            array = array[::-1]
        elif name == "rot90":
            array = np.rot90(array, k, axes=(0, 1))
        elif name == "transpose":
            array = np.swapaxes(array, 0, 1)
        else:
            raise ValueError(f"unknown geometric op {name!r}")
    return np.ascontiguousarray(array)


def sin4_sign(ops: list[tuple[str, int]]) -> int:
    """Factor of sin 4θ under ops (docs/METHOD.md §3.5): −1 per flip or transpose, +1 per rotation; cos 4θ keeps."""
    return -1 if sum(name != "rot90" for name, _ in ops) % 2 else 1


def photometric(image: np.ndarray, rng: np.random.Generator, limit: float) -> np.ndarray:
    """RGB brightness/contrast change: image · a + b · 255, a in [1 − limit, 1 + limit] and b in [−limit, limit]."""
    a, b = rng.uniform(1 - limit, 1 + limit), rng.uniform(-limit, limit)
    return np.clip(np.rint(image.astype(np.float32) * a + b * 255), 0, 255).astype(np.uint8)


def pad(array: np.ndarray, size: int, value=None) -> np.ndarray:
    """Pad the first two axes at the bottom and right to size: reflect when value is None, else a constant."""
    height, width = array.shape[:2]
    if height > size or width > size:
        raise ValueError(f"a {height} × {width} array does not fit in {size} × {size}")
    widths = [(0, size - height), (0, size - width)] + [(0, 0)] * (array.ndim - 2)
    if value is None:
        return np.pad(array, widths, mode="reflect")
    return np.pad(array, widths, mode="constant", constant_values=value)


@dataclass
class Sample:
    image: np.ndarray  # (S, S, 3) uint8 RGB
    priors: np.ndarray | None  # (S, S, k) uint8 prior channels, or None for RGB-only models
    targets: Targets  # padded as described in the module docstring


def make_sample(
    image: np.ndarray,
    priors: np.ndarray | None,
    instance_map: np.ndarray,
    cfg,
    rng: np.random.Generator | None = None,
) -> Sample:
    """One model input with its targets; rng None leaves the tile unaugmented (validation and test)."""
    if rng is not None:
        ops = random_geometry(rng, cfg.augment)
        image, instance_map = apply_geometry(image, ops), apply_geometry(instance_map, ops)
        priors = None if priors is None else apply_geometry(priors, ops)
        if rng.random() < cfg.augment.p_photometric:
            image = photometric(image, rng, cfg.augment.photometric_limit)
    targets = targets_from_config(instance_map, cfg.labels)
    size = int(cfg.data.pad_to)
    return Sample(
        image=pad(image, size),
        priors=None if priors is None else pad(priors, size),
        targets=Targets(
            classes=pad(targets.classes, size, int(cfg.data.ignore_index)),
            sin4=pad(targets.sin4, size, 0.0),
            cos4=pad(targets.cos4, size, 0.0),
            valid=pad(targets.valid, size, False),
            weight=pad(targets.weight, size, 0.0),
        ),
    )
