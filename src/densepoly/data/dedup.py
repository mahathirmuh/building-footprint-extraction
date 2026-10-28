"""CrowdAI deduplication with p-hash under rotations and flips (docs/DATA.md §2.2).

Own implementation of the logic of CrowdAI_Hash_and_search (Adimoolam et al., MIT licence): every image gets the
``imagehash.phash`` of six variants (original, rotations by 90/180/270°, horizontal and vertical flip), and two images
belong to one cluster when any variant hash of one equals any variant hash of the other. Matching variant against
variant also catches transposed copies, so all 8 rotations/flips of a square tile end up in the same cluster.
"""

from __future__ import annotations

import multiprocessing
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image, ImageDraw

from densepoly.utils.io import read_json, write_json

VARIANTS = ("original", "rot90", "rot180", "rot270", "flip_h", "flip_v")
_TRANSPOSES = (
    None,
    Image.Transpose.ROTATE_90,
    Image.Transpose.ROTATE_180,
    Image.Transpose.ROTATE_270,
    Image.Transpose.FLIP_LEFT_RIGHT,
    Image.Transpose.FLIP_TOP_BOTTOM,
)

# Verified on 28 Oct 2026 from the CrowdAI_Hash_and_search files and README (CLAUDE.md, "Status").
REFERENCE = {
    "train_images": 280741,
    "val_images": 60317,
    "train_unique_original_phash": 114548,
    "val_without_exact_duplicate_in_train": 3949,
    "train_unique_after_augmentation": "about 30k (repo README: about 250k of 280k are exact or augmented duplicates)",
}


def variant_hashes(image: Image.Image) -> tuple[str, ...]:
    """p-hash (64-bit hex) of the six variants, in the order of VARIANTS."""
    gray = image.convert("L")  # phash works on grayscale; converting first is exact since transposes only move pixels
    return tuple(str(imagehash.phash(gray if t is None else gray.transpose(t))) for t in _TRANSPOSES)


def hash_file(path: Path) -> tuple[str, ...]:
    with Image.open(path) as image:
        return variant_hashes(image)


def hash_images(files: Sequence[Path], cache_dir: Path, workers: int = 1, chunk: int = 5000) -> list[tuple[str, ...]]:
    """Variant hashes of every file, in order. Each chunk of `chunk` files is cached as JSON under cache_dir, so an
    interrupted run resumes where it stopped; a chunk whose file list changed is recomputed."""
    out: list[tuple[str, ...]] = []
    pool = None
    try:
        for k, start in enumerate(range(0, len(files), chunk)):
            part = list(files[start : start + chunk])
            names = [p.name for p in part]
            cache = cache_dir / f"chunk_{k:05d}.json"
            if cache.exists():
                cached = read_json(cache)
                if cached["images"] == names:
                    out.extend(tuple(h) for h in cached["hashes"])
                    continue
            if workers > 1 and pool is None:
                pool = multiprocessing.get_context("spawn").Pool(workers)
            hashes = pool.map(hash_file, part, chunksize=32) if pool else [hash_file(p) for p in part]
            write_json(cache, {"images": names, "hashes": [list(h) for h in hashes]}, indent=None)
            out.extend(hashes)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return out


def benchmark(files: Sequence[Path], n_total: int, workers: int) -> dict:
    """Time the hashing of `files` on one core and project it to n_total images."""
    start = time.perf_counter()
    for path in files:
        hash_file(path)
    per_image = (time.perf_counter() - start) / max(len(files), 1)
    core_hours = per_image * n_total / 3600
    return {
        "images_timed": len(files),
        "ms_per_image": round(per_image * 1000, 2),
        "projected_core_hours": round(core_hours, 2),
        "projected_wall_minutes_ideal": round(core_hours * 60 / workers, 1),
        "workers": workers,
    }


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:
            self.parent[i], i = root, self.parent[i]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def cluster(names: Sequence[str], hashes: Sequence[Sequence[str]]) -> dict[str, list[str]]:
    """Group images that share any variant hash: {representative (smallest file name): sorted members}."""
    uf = _UnionFind(len(names))
    first: dict[str, int] = {}
    for i in sorted(range(len(names)), key=lambda i: names[i]):
        for h in hashes[i]:
            j = first.setdefault(h, i)
            if j != i:
                uf.union(i, j)
    groups: dict[int, list[str]] = defaultdict(list)
    for i, name in enumerate(names):
        groups[uf.find(i)].append(name)
    return dict(sorted((min(m), sorted(m)) for m in groups.values()))


def split_train(
    clusters: dict[str, list[str]],
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    tune_sub_size: int = 1000,
    seed: int = 0,
) -> dict[str, list[str]]:
    """One representative per cluster. The clusters are shuffled once (seed): the first `val_fraction` of them form
    val_tune, the next `test_fraction` the held-out test, the rest train_dedup. val_tune is then split (same seed)
    into val_tune_sub (tune_sub_size images) and val_select."""
    reps = sorted(clusters)
    order = np.random.default_rng(seed).permutation(len(reps)).tolist()
    n_val, n_test = int(round(val_fraction * len(reps))), int(round(test_fraction * len(reps)))
    tune, test = set(order[:n_val]), set(order[n_val : n_val + n_test])
    val_tune = [r for i, r in enumerate(reps) if i in tune]
    sub = set(np.random.default_rng(seed).permutation(len(val_tune))[:tune_sub_size].tolist())
    return {
        "train_dedup": [r for i, r in enumerate(reps) if i not in tune and i not in test],
        "val_tune": val_tune,
        "val_tune_sub": [r for i, r in enumerate(val_tune) if i in sub],
        "val_select": [r for i, r in enumerate(val_tune) if i not in sub],
        "test": [r for i, r in enumerate(reps) if i in test],
    }


@dataclass
class DedupResult:
    splits: dict[str, list[str]]  # train_dedup, val_tune, val_tune_sub, val_select, test, val_clean
    train_clusters: dict[str, list[str]]
    val_clean_clusters: dict[str, list[str]]
    report: dict


def build_splits(
    train_names: Sequence[str],
    train_hashes: Sequence[Sequence[str]],
    val_names: Sequence[str],
    val_hashes: Sequence[Sequence[str]],
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    tune_sub_size: int = 1000,
    seed: int = 0,
) -> DedupResult:
    """All splits of docs/DATA.md §2.2 plus a report comparing our counts with the verified reference numbers.

    val_clean keeps val images none of whose six hashes occurs among any train variant hash, then keeps one
    representative per duplicate cluster inside val.
    """
    train_clusters = cluster(train_names, train_hashes)
    splits = split_train(train_clusters, val_fraction, test_fraction, tune_sub_size, seed)
    train_any = {h for hs in train_hashes for h in hs}
    train_original = {hs[0] for hs in train_hashes}
    pairs = list(zip(val_names, val_hashes, strict=True))
    clean = [(n, hs) for n, hs in pairs if train_any.isdisjoint(hs)]
    val_clean_clusters = cluster([n for n, _ in clean], [hs for _, hs in clean])
    splits["val_clean"] = sorted(val_clean_clusters)

    sizes = np.array([len(m) for m in train_clusters.values()])
    largest = sorted(train_clusters.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:10]
    report = {
        "train": {
            "images": len(train_names),
            "unique_original_hashes": len(train_original),
            "clusters": len(train_clusters),
            "singleton_clusters": int((sizes == 1).sum()),
            "cluster_size_mean": round(float(sizes.mean()), 3),
            "cluster_size_max": int(sizes.max()),
            "largest_clusters": [[rep, len(members)] for rep, members in largest],
        },
        "val": {
            "images": len(val_names),
            "original_hash_not_in_train_originals": sum(hs[0] not in train_original for _, hs in pairs),
            "no_variant_hash_in_train": len(clean),
            "clean_clusters": len(val_clean_clusters),
        },
        "splits": {name: len(items) for name, items in splits.items()},
        "params": {
            "val_fraction": val_fraction,
            "test_fraction": test_fraction,
            "tune_sub_size": tune_sub_size,
            "seed": seed,
            "variants": VARIANTS,
        },
        "reference": REFERENCE,
    }
    return DedupResult(splits, train_clusters, val_clean_clusters, report)


def cluster_mosaic(
    clusters: dict[str, list[str]],
    image_dir: Path,
    n_largest: int = 10,
    n_random: int = 40,
    max_members: int = 6,
    thumb: int = 64,
    seed: int = 0,
) -> Image.Image:
    """Thumbnail grid for visual QA, one row per multi-image cluster (representative first): the largest clusters,
    where p-hash collisions between near-uniform tiles would show up, followed by random clusters."""
    multi = sorted((rep for rep, m in clusters.items() if len(m) > 1), key=lambda r: (-len(clusters[r]), r))
    rest = multi[n_largest:]
    picked = np.random.default_rng(seed).choice(len(rest), size=min(n_random, len(rest)), replace=False) if rest else []
    rows = multi[:n_largest] + [rest[i] for i in sorted(picked)]
    label, step = 150, thumb + 2
    canvas = Image.new("RGB", (label + max_members * step, max(len(rows), 1) * step), "white")
    draw = ImageDraw.Draw(canvas)
    for r, rep in enumerate(rows):
        draw.text((2, r * step + 2), f"{rep}\n{len(clusters[rep])} images", fill="black")
        for c, name in enumerate(clusters[rep][:max_members]):
            with Image.open(image_dir / name) as image:
                canvas.paste(image.convert("RGB").resize((thumb, thumb)), (label + c * step, r * step))
    return canvas
