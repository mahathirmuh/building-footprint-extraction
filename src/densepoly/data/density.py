"""Density strata of a split, contribution N2 (docs/DATA.md §4).

Main index: n = number of non-empty GT instances in a tile. The thresholds are the tertiles of n over the test tiles
with n >= 1; strata are kosong (n = 0), jarang (n <= q33), sedang (q33 < n <= q67) and padat (n > q67). val_tune
reuses the test thresholds. Secondary index (robustness, appendix): the touching ratio, i.e. the fraction of a tile's
instances that own contact pixels (docs/METHOD.md §3.2). Both depend on GT only; the file is frozen before any model
is evaluated.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from densepoly.labels.contact import contact_mask

STRATA = ("kosong", "jarang", "sedang", "padat")
DEFINITION = {
    "main": "n = number of non-empty GT instances per tile; tertiles of n over test tiles with n >= 1; "
    "kosong: n = 0, jarang: n <= q33, sedang: q33 < n <= q67, padat: n > q67",
    "secondary": "touching ratio = fraction of a tile's instances with contact pixels (docs/METHOD.md §3.2); "
    "tertiles over test tiles with at least one building; kosong: no building",
}


def touching_ratio(instance_map: np.ndarray, radius: int) -> float | None:
    """Fraction of the tile's instances owning at least one contact pixel; None for a tile without buildings."""
    ids = np.unique(instance_map[instance_map > 0])
    if ids.size == 0:
        return None
    return np.unique(instance_map[contact_mask(instance_map, radius)]).size / ids.size


def tertiles(values: Iterable[float]) -> tuple[float, float]:
    """q33 and q67 (numpy's default linear interpolation)."""
    v = np.fromiter(values, float)
    if v.size == 0:
        raise ValueError("no tile with buildings to compute tertiles from")
    q33, q67 = np.quantile(v, [1 / 3, 2 / 3])
    return float(q33), float(q67)


def assign(values: dict[int, float | None], q33: float, q67: float) -> dict[str, list[int]]:
    """Image ids per stratum; a value of None (tile without buildings) goes to kosong."""
    strata: dict[str, list[int]] = {name: [] for name in STRATA}
    for image_id, v in sorted(values.items()):
        name = "kosong" if v is None else "jarang" if v <= q33 else "sedang" if v <= q67 else "padat"
        strata[name].append(image_id)
    return strata


def strata_record(
    counts: dict[int, int], ratios: dict[int, float | None], thresholds: dict[str, list[float]] | None = None
) -> dict:
    """Strata of one split under both indices. thresholds=None derives them from these tiles (the test split);
    otherwise the given {"main": [q33, q67], "secondary": [q33, q67]} are reused (val_tune)."""
    main = {i: (float(n) if n > 0 else None) for i, n in counts.items()}
    if thresholds is None:
        thresholds = {
            "main": list(tertiles(v for v in main.values() if v is not None)),
            "secondary": list(tertiles(v for v in ratios.values() if v is not None)),
        }
    record = {"definition": DEFINITION, "thresholds": thresholds}
    for index, values in (("main", main), ("secondary", ratios)):
        strata = assign(values, *thresholds[index])
        record[index] = {"sizes": {k: len(v) for k, v in strata.items()}, "strata": strata}
    return record
