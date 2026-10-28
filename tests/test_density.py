"""Density strata (docs/DATA.md §4) on counts with a known answer."""

import numpy as np
import pytest

from densepoly.data import density
from tests.fixtures.synthetic import make_scene


def test_tertiles_and_strata_on_known_counts():
    counts = dict(enumerate([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))  # image id -> number of buildings
    assert density.tertiles(n for n in counts.values() if n > 0) == pytest.approx((11 / 3, 19 / 3))
    record = density.strata_record(counts, {i: (None if n == 0 else 0.5) for i, n in counts.items()})
    main = record["main"]
    assert main["sizes"] == {"kosong": 2, "jarang": 3, "sedang": 3, "padat": 3}
    assert main["strata"]["kosong"] == [0, 1] and main["strata"]["padat"] == [8, 9, 10]


def test_every_tile_lands_in_exactly_one_stratum():
    rng = np.random.default_rng(0)
    counts = {i: int(n) for i, n in enumerate(rng.poisson(6, 500))}
    ratios = {i: (None if n == 0 else float(rng.random())) for i, n in counts.items()}
    record = density.strata_record(counts, ratios)
    for index in ("main", "secondary"):
        members = [i for ids in record[index]["strata"].values() for i in ids]
        assert sorted(members) == sorted(counts)


def test_val_tune_reuses_the_test_thresholds():
    thresholds = {"main": [3.0, 6.0], "secondary": [0.2, 0.6]}
    record = density.strata_record({1: 2, 2: 50, 3: 0}, {1: 0.0, 2: 1.0, 3: None}, thresholds)
    assert record["thresholds"] == thresholds
    assert record["main"]["strata"] == {"kosong": [3], "jarang": [1], "sedang": [], "padat": [2]}
    assert record["secondary"]["strata"] == {"kosong": [3], "jarang": [1], "sedang": [], "padat": [2]}


def test_touching_ratio_on_a_synthetic_scene():
    """Only touching-row houses, and gap-row houses whose 1 px gap is within r = 2, own contact pixels."""
    scene = make_scene(0)
    touching = {b.id for b in scene.buildings if b.neighbors}
    touching |= {x for a, c, width in scene.gap_pairs if width + 1 <= 2 for x in (a, c)}
    assert density.touching_ratio(scene.instance_map, 2) == pytest.approx(len(touching) / len(scene.buildings))
    assert density.touching_ratio(np.zeros((8, 8), np.uint16), 2) is None
