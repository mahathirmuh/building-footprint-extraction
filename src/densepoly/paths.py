"""Project filesystem layout (CLAUDE.md "Data"; docs/DATA.md §1; docs/EVALUATION.md §1).

Datasets and outputs live outside git and are located through environment variables: ``DENSEPOLY_DATA``
(default ``<repo>/data``) and ``DENSEPOLY_OUT`` (default ``<repo>/outputs``). Defaults are anchored to the
repository root rather than the working directory, so scripts resolve the same folders wherever they are
launched. Version-controlled folders (``configs/``, ``splits/``) always come from the repository.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATASETS = ("crowdai", "whu")
# val_tune_sub and val_select are the two parts of val_tune. CrowdAI only: val_clean (val tiles without any duplicate
# in train) compares with HiSup's released predictions; val_full (the whole, leaky val) only reproduces HiSup.
SPLITS = ("train_dedup", "val_tune", "val_tune_sub", "val_select", "test", "val_clean", "val_full")
CROWDAI_ONLY = ("val_clean", "val_full")


def _env_dir(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else default


def _check(dataset: str, split: str | None = None) -> None:
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")
    if split is not None and split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    if split in CROWDAI_ONLY and dataset != "crowdai":
        raise ValueError(f"split {split!r} exists only for crowdai")


def data_root() -> Path:
    """Datasets and caches: $DENSEPOLY_DATA, default <repo>/data."""
    return _env_dir("DENSEPOLY_DATA", REPO_ROOT / "data")


def out_root() -> Path:
    """Generated outputs: $DENSEPOLY_OUT, default <repo>/outputs."""
    return _env_dir("DENSEPOLY_OUT", REPO_ROOT / "outputs")


def configs_dir() -> Path:
    return REPO_ROOT / "configs"


def splits_dir() -> Path:
    return REPO_ROOT / "splits"


def split_list(dataset: str, split: str, root: Path | None = None) -> Path:
    """Text file with one image file name per line (docs/DATA.md §6)."""
    _check(dataset, split)
    return (root or splits_dir()) / f"{dataset}_{split}.txt"


def density_file(dataset: str, split: str, root: Path | None = None) -> Path:
    """Frozen density strata of a split (docs/DATA.md §4)."""
    _check(dataset, split)
    return (root or splits_dir()) / f"density_{dataset}_{split}.json"


def run_name(seed: int | None) -> str:
    """Run folder name: ``seed<k>`` for trained models, ``ref`` for seedless runs (B0, oracle, HiSup release)."""
    return "ref" if seed is None else f"seed{seed}"


def dataset_dir(dataset: str) -> Path:
    _check(dataset)
    return data_root() / dataset


def source_folder(dataset: str, split: str) -> str:
    """Folder holding a split's images: CrowdAI download train/ or val/ (test is held out from train; val_clean and
    val_full come from val); WHU protocol A conversion train/, val/ (val_tune and its parts) or test/."""
    _check(dataset, split)
    if dataset == "crowdai":
        return "val" if split in CROWDAI_ONLY else "train"
    return {"train_dedup": "train", "test": "test"}.get(split, "val")


def _layout_root(dataset: str) -> Path:
    return dataset_dir("whu") / "pix2poly_224" if dataset == "whu" else dataset_dir(dataset)


def annotation_file(dataset: str, split: str) -> Path:
    """COCO ground truth that holds a split's images (docs/DATA.md §1)."""
    return _layout_root(dataset) / source_folder(dataset, split) / "annotation.json"


def image_dir(dataset: str, split: str) -> Path:
    return _layout_root(dataset) / source_folder(dataset, split) / "images"


def instance_dir(dataset: str, split: str) -> Path:
    """Folder of a split's cached instance maps (one uint16 PNG per image plus index.csv)."""
    _check(dataset, split)
    return data_root() / "instances" / dataset / split


def instance_cache(dataset: str, split: str, image_id: int | str) -> Path:
    """uint16 PNG instance map of one image."""
    return instance_dir(dataset, split) / f"{image_id}.png"


def prior_cache(dataset: str, split: str, image_id: int | str) -> Path:
    """uint8 PNG prior of one image (R = MBI, G = gap map, B = 0)."""
    _check(dataset, split)
    return data_root() / "priors" / dataset / split / f"{image_id}.png"


def run_dir(dataset: str, exp: str, run: str) -> Path:
    """Training run folder: checkpoints, logs and the frozen config."""
    _check(dataset)
    return out_root() / "runs" / dataset / exp / run


def pred_file(dataset: str, exp: str, run: str, split: str) -> Path:
    _check(dataset, split)
    return out_root() / "preds" / dataset / exp / run / f"{split}.json"


def metrics_file(dataset: str, exp: str, run: str, split: str) -> Path:
    _check(dataset, split)
    return out_root() / "metrics" / dataset / exp / run / f"{split}.json"


def gt_file(dataset: str, split: str) -> Path:
    """COCO ground truth restricted to one split (all of its images, including empty ones)."""
    _check(dataset, split)
    return out_root() / "gt" / dataset / f"{split}.json"
