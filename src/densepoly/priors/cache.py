"""Prior cache (docs/METHOD.md §2.4, docs/DATA.md §5).

The MBI and the gap map are computed once from the original images and stored as uint8 PNG (R = MBI, G = gap map,
B = 0) after scaling every channel by its 99.5th percentile q over 500 random train units (seed 0):
uint8(clip(v / q, 0, 1) · 255). q and the mean/std of the scaled channels (v / 255) are kept in
configs/data/<dataset>_prior_stats.json; the training loader normalises a channel as (v / 255 − mean) / std.

A unit is one CrowdAI image (key: image id) or one official WHU tile (key: tile stem, 1024²); WHU patches are cropped
from the cached tile at the offsets of patches.csv, so that MBI scales up to ±120 px see the whole tile rather than
a 224 px patch, and overlapping patches are not computed twice.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from densepoly import paths
from densepoly.priors.gap import gap_map
from densepoly.priors.mbi import brightness, mbi
from densepoly.utils.io import read_json, read_lines, read_png, write_png

CHANNELS = ("mbi", "gap")
PERCENTILE = 99.5
STATS_UNITS = 500  # random train units for the scale and the normalisation statistics
STATS_PIXELS_PER_UNIT = 20_000  # random pixels per unit: all pixels of 500 WHU tiles would not fit in memory


@dataclass(frozen=True)
class Unit:
    key: str  # cache file stem: CrowdAI image id or WHU tile stem
    image: Path  # RGB source image


def units(dataset: str, split: str, splits_dir: Path | None = None) -> list[Unit]:
    """Units whose priors a split needs, in a stable order."""
    names = read_lines(paths.split_list(dataset, split, splits_dir))
    if dataset == "crowdai":
        with (paths.instance_dir(dataset, split) / "index.csv").open(encoding="utf-8") as f:
            image_ids = {row["file_name"]: row["image_id"] for row in csv.DictReader(f)}
        folder = paths.image_dir(dataset, split)
        return [Unit(image_ids[name], folder / name) for name in names]
    wanted = set(names)
    with paths.patch_index(dataset, split).open(encoding="utf-8") as f:
        stems = sorted({row["tile"] for row in csv.DictReader(f) if row["file_name"] in wanted})
    folder = paths.dataset_dir(dataset) / "raw_0.2" / paths.source_folder(dataset, split) / "image"
    return [Unit(stem, folder / f"{stem}.tif") for stem in stems]


def read_rgb(path: Path) -> np.ndarray:
    """(H, W, 3) uint8 RGB of a JPEG (CrowdAI) or an LZW TIFF (WHU)."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def compute_priors(image: np.ndarray, priors) -> np.ndarray:
    """(H, W, 2) float32 raw priors [MBI, gap] of an RGB image; priors is the config section (configs/base.yaml)."""
    b = brightness(image)
    return np.stack(
        [mbi(b, list(priors["mbi_scales_px"])), gap_map(b, int(priors["gap_se_length_px"]), priors["gap_mode"])],
        axis=-1,
    )


def quantize(raw: np.ndarray, q) -> np.ndarray:
    """uint8(clip(v / q, 0, 1) · 255) per channel."""
    scale = np.maximum(np.asarray(q, np.float32), 1e-12)
    return np.rint(np.clip(raw / scale, 0.0, 1.0) * 255.0).astype(np.uint8)


def write_prior(path: Path, prior: np.ndarray) -> None:
    """Store (H, W, 2) uint8 [MBI, gap] as a PNG with R = MBI, G = gap, B = 0 (cv2 writes channels as B, G, R)."""
    write_png(path, np.dstack([np.zeros_like(prior[..., 0]), prior[..., 1], prior[..., 0]]))


def read_prior(path: Path) -> np.ndarray:
    """(H, W, 2) uint8 [MBI, gap] from a prior PNG."""
    bgr = read_png(path)
    return np.dstack([bgr[..., 2], bgr[..., 1]])


def stats_file(dataset: str) -> Path:
    return paths.configs_dir() / "data" / f"{dataset}_prior_stats.json"


def load_stats(dataset: str) -> dict:
    """Frozen scale and normalisation statistics of a dataset (written by scripts/precompute_priors.py --stats)."""
    return read_json(stats_file(dataset))


def statistics(samples: np.ndarray) -> dict:
    """q (99.5th percentile) per channel of raw prior samples (N, 2), and mean/std of the scaled values v / 255."""
    q = np.percentile(samples, PERCENTILE, axis=0)
    scaled = quantize(samples, q).astype(np.float64) / 255.0
    return {
        "channels": list(CHANNELS),
        "q": [float(v) for v in q],
        "mean": [float(v) for v in scaled.mean(axis=0)],
        "std": [float(v) for v in scaled.std(axis=0)],
    }


def normalize(prior: np.ndarray, stats: dict) -> np.ndarray:
    """(H, W, 2) float32 model input from uint8 priors: (v / 255 − mean) / std per channel."""
    mean, std = np.asarray(stats["mean"], np.float32), np.asarray(stats["std"], np.float32)
    return (prior.astype(np.float32) / 255.0 - mean) / std


def sample_unit(unit: Unit, priors: dict, n: int, seed: int) -> tuple[np.ndarray, float]:
    """Raw prior values at n random pixels of one unit, and the seconds spent computing its priors."""
    start = time.perf_counter()
    raw = compute_priors(read_rgb(unit.image), priors).reshape(-1, len(CHANNELS))
    seconds = time.perf_counter() - start
    picked = np.random.default_rng(seed).choice(len(raw), size=min(n, len(raw)), replace=False)
    return raw[picked], seconds


def cache_unit(unit: Unit, target: Path, priors: dict, q: list[float]) -> float:
    """Compute, scale and store the priors of one unit; return the seconds spent computing them."""
    start = time.perf_counter()
    raw = compute_priors(read_rgb(unit.image), priors)
    seconds = time.perf_counter() - start
    write_prior(target, quantize(raw, q))
    return seconds
