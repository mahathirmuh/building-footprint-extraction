"""Density strata of a split, contribution N2 (docs/DATA.md §4).

Main index: n = number of non-empty GT instances in a tile. The thresholds are the tertiles of n over the test tiles
with n >= 1; strata are kosong (n = 0), jarang (n <= q33), sedang (q33 < n <= q67) and padat (n > q67). val_tune and
val_clean reuse the test thresholds. Secondary index (robustness, appendix): whether the tile has touching buildings,
i.e. at least one instance owning contact pixels (docs/METHOD.md §3.2). Tertiles of the touching ratio were planned,
but 95% of the CrowdAI test tiles have no touching building, so both tertiles are 0 (docs/progress.md, 28 Oct 2026).
Both indices depend on GT only; the file is frozen before any model is evaluated.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from densepoly.labels.contact import contact_mask

STRATA = ("kosong", "jarang", "sedang", "padat")
TOUCH_STRATA = ("kosong", "tidak_bersentuhan", "bersentuhan")
DEFINITION = {
    "main": "n = number of non-empty GT instances per tile; tertiles of n over test tiles with n >= 1; "
    "kosong: n = 0, jarang: n <= q33, sedang: q33 < n <= q67, padat: n > q67",
    "secondary": "bersentuhan: at least one GT instance owns contact pixels, i.e. another building lies within the "
    "contact radius (docs/METHOD.md §3.2); tidak_bersentuhan: none does; kosong: no building",
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


def assign_touching(ratios: dict[int, float | None]) -> dict[str, list[int]]:
    """Image ids per secondary stratum: kosong (no building), tidak_bersentuhan (ratio 0) or bersentuhan (ratio > 0)."""
    strata: dict[str, list[int]] = {name: [] for name in TOUCH_STRATA}
    for image_id, v in sorted(ratios.items()):
        strata["kosong" if v is None else "bersentuhan" if v > 0 else "tidak_bersentuhan"].append(image_id)
    return strata


def strata_record(
    counts: dict[int, int], ratios: dict[int, float | None], thresholds: dict[str, list[float]] | None = None
) -> dict:
    """Strata of one split under both indices. thresholds=None derives the main tertiles from these tiles (the test
    split); otherwise the given {"main": [q33, q67]} are reused (val_tune, val_clean). The secondary index has no
    threshold."""
    main = {i: (float(n) if n > 0 else None) for i, n in counts.items()}
    q33, q67 = tertiles(v for v in main.values() if v is not None) if thresholds is None else thresholds["main"]
    record = {"definition": DEFINITION, "thresholds": {"main": [q33, q67]}}
    for index, strata in (("main", assign(main, q33, q67)), ("secondary", assign_touching(ratios))):
        record[index] = {"sizes": {k: len(v) for k, v in strata.items()}, "strata": strata}
    return record
