"""Deduplication (docs/DATA.md §2.2): variant hashes, clustering, splits and the clean test."""

import numpy as np
import pytest
from PIL import Image

from densepoly.data import dedup
from tests.fixtures.synthetic import make_scene

T = Image.Transpose
DIHEDRAL = [
    None,
    T.ROTATE_90,
    T.ROTATE_180,
    T.ROTATE_270,
    T.FLIP_LEFT_RIGHT,
    T.FLIP_TOP_BOTTOM,
    T.TRANSPOSE,
    T.TRANSVERSE,
]


@pytest.fixture(scope="module")
def images():
    return {s: Image.fromarray(make_scene(s).image) for s in range(4)}


def test_all_eight_rotations_and_flips_share_a_variant_hash(images):
    base = dedup.variant_hashes(images[0])
    assert len(base) == len(dedup.VARIANTS)
    for op in DIHEDRAL:
        copy = images[0] if op is None else images[0].transpose(op)
        assert set(base) & set(dedup.variant_hashes(copy)), op


def test_different_scenes_share_no_hash(images):
    sets = [set(dedup.variant_hashes(image)) for image in images.values()]
    assert all(a.isdisjoint(b) for i, a in enumerate(sets) for b in sets[i + 1 :])


def test_cluster_groups_copies_under_the_smallest_name(images):
    names = ["b.png", "a_rot.png", "a.png", "c.png", "b_flip.png"]
    pictures = [
        images[1],
        images[0].transpose(T.ROTATE_90),
        images[0],
        images[2],
        images[1].transpose(T.FLIP_TOP_BOTTOM),
    ]
    clusters = dedup.cluster(names, [dedup.variant_hashes(p) for p in pictures])
    assert clusters == {"a.png": ["a.png", "a_rot.png"], "b.png": ["b.png", "b_flip.png"], "c.png": ["c.png"]}


def test_cluster_is_transitive():
    clusters = dedup.cluster(["a", "b", "c", "d"], [["h1", "h2"], ["h2", "h3"], ["h3", "h4"], ["h5"]])
    assert clusters == {"a": ["a", "b", "c"], "d": ["d"]}


def synthetic_clusters(n=200, seed=0):
    rng = np.random.default_rng(seed)
    clusters = {}
    for k in range(n):
        members = [f"img{k:04d}_{j}" for j in range(int(rng.integers(1, 6)))]
        clusters[members[0]] = members
    return clusters


def test_split_train_is_disjoint_complete_and_deterministic():
    clusters = synthetic_clusters()
    splits = dedup.split_train(clusters, val_fraction=0.1, tune_sub_size=8, seed=0)
    assert splits == dedup.split_train(clusters, val_fraction=0.1, tune_sub_size=8, seed=0)
    train, tune = set(splits["train_dedup"]), set(splits["val_tune"])
    assert len(tune) == 20 and not train & tune and train | tune == set(clusters)
    sub, select = set(splits["val_tune_sub"]), set(splits["val_select"])
    assert len(sub) == 8 and not sub & select and sub | select == tune
    train_members = {m for rep in train for m in clusters[rep]}
    assert train_members.isdisjoint(m for rep in tune for m in clusters[rep])  # no cluster spans two splits
    assert dedup.split_train(clusters, seed=1)["val_tune"] != splits["val_tune"]


def test_build_splits_keeps_only_clean_and_deduplicated_val_images():
    train_names, train_hashes = ["t1", "t2"], [["a", "b"], ["c", "d"]]
    val_names = ["v1", "v2", "v3", "v4"]
    val_hashes = [["x", "b"], ["e", "f"], ["g", "e"], ["h", "i"]]  # v1 leaks from t1; v3 copies v2
    result = dedup.build_splits(train_names, train_hashes, val_names, val_hashes, tune_sub_size=1)
    assert result.splits["test"] == ["v2", "v4"]
    assert result.test_clusters == {"v2": ["v2", "v3"], "v4": ["v4"]}
    assert result.report["val"]["no_variant_hash_in_train"] == 3
    assert result.report["val"]["original_hash_not_in_train_originals"] == 4
    assert result.report["splits"]["test"] == 2


def save_images(folder, images):
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for s, image in images.items():
        image.save(folder / f"{s}.png")
        files.append(folder / f"{s}.png")
    return files


def test_hash_images_caches_chunks_and_resumes_without_the_images(tmp_path, images):
    files = save_images(tmp_path / "img", images)
    cache = tmp_path / "cache"
    first = dedup.hash_images(files, cache, workers=1, chunk=3)
    assert first == [dedup.variant_hashes(image) for image in images.values()]
    assert len(list(cache.glob("chunk_*.json"))) == 2
    for path in files:
        path.unlink()
    assert dedup.hash_images(files, cache, workers=1, chunk=3) == first


def test_parallel_hashing_matches_serial(tmp_path, images):
    files = save_images(tmp_path / "img", images)
    serial = dedup.hash_images(files, tmp_path / "serial", workers=1)
    assert dedup.hash_images(files, tmp_path / "parallel", workers=2) == serial


def test_cluster_mosaic_has_one_row_per_multi_image_cluster(tmp_path, images):
    for name, image in {"a.png": images[0], "a2.png": images[0], "b.png": images[1]}.items():
        image.save(tmp_path / name)
    clusters = {"a.png": ["a.png", "a2.png"], "b.png": ["b.png"]}
    mosaic = dedup.cluster_mosaic(clusters, tmp_path, thumb=32, max_members=3)
    assert mosaic.size == (150 + 3 * 34, 34)
