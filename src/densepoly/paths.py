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
# WHU has two ground truths on the same patches (docs/DATA.md §3.2): annotation.json keeps the official buildings
# apart; variant "p2p" (annotation_p2p.json) is Pix2Poly's, only for comparing with the Pix2Poly paper.
GT_VARIANTS = {"whu": ("p2p",)}


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
    val_full come from val); WHU option R conversion train/, val/ (val_tune and its parts) or test/."""
    _check(dataset, split)
    if dataset == "crowdai":
        return "val" if split in CROWDAI_ONLY else "train"
    return {"train_dedup": "train", "test": "test"}.get(split, "val")


def _layout_root(dataset: str) -> Path:
    return dataset_dir("whu") / "pix2poly_224" if dataset == "whu" else dataset_dir(dataset)


def _check_variant(dataset: str, variant: str | None) -> None:
    if variant is not None and variant not in GT_VARIANTS.get(dataset, ()):
        raise ValueError(f"unknown ground truth variant {variant!r} for {dataset}")


def annotation_file(dataset: str, split: str, variant: str | None = None) -> Path:
    """COCO ground truth that holds a split's images (docs/DATA.md §1); variant selects another ground truth of the
    same images (GT_VARIANTS)."""
    _check_variant(dataset, variant)
    name = "annotation.json" if variant is None else f"annotation_{variant}.json"
    return _layout_root(dataset) / source_folder(dataset, split) / name


def patch_index(dataset: str, split: str) -> Path:
    """WHU only: patches.csv with the official tile and offset of every patch (patches are not stored as images)."""
    if dataset != "whu":
        raise ValueError("only WHU patches are cut from larger tiles")
    return _layout_root(dataset) / source_folder(dataset, split) / "patches.csv"


def image_dir(dataset: str, split: str) -> Path:
    return _layout_root(dataset) / source_folder(dataset, split) / "images"


def cache_split(split: str) -> str:
    """Split whose cache folder holds a split's files: val_tune_sub and val_select are parts of val_tune."""
    return {"val_tune_sub": "val_tune", "val_select": "val_tune"}.get(split, split)


def instance_dir(dataset: str, split: str) -> Path:
    """Folder of a split's cached instance maps (one uint16 PNG per image plus index.csv)."""
    _check(dataset, split)
    return data_root() / "instances" / dataset / cache_split(split)


def instance_cache(dataset: str, split: str, image_id: int | str) -> Path:
    """uint16 PNG instance map of one image."""
    return instance_dir(dataset, split) / f"{image_id}.png"


def prior_cache(dataset: str, split: str, key: int | str) -> Path:
    """uint8 PNG prior (R = MBI, G = gap map, B = 0) of one CrowdAI image id or one official WHU tile stem (WHU priors
    are computed on the official 1024² tiles and cropped to the patches, docs/DATA.md §5)."""
    _check(dataset, split)
    return data_root() / "priors" / dataset / cache_split(split) / f"{key}.png"


def run_dir(dataset: str, exp: str, run: str) -> Path:
    """Training run folder: checkpoints, logs and the frozen config."""
    _check(dataset)
    return out_root() / "runs" / dataset / exp / run


def pred_file(dataset: str, exp: str, run: str, split: str) -> Path:
    _check(dataset, split)
    return out_root() / "preds" / dataset / exp / run / f"{split}.json"


def metrics_file(dataset: str, exp: str, run: str, split: str, variant: str | None = None) -> Path:
    """Metrics of one evaluation (docs/EVALUATION.md §5); variant: the GT variant evaluated against (gt_file)."""
    return metrics_artifact(dataset, exp, run, split, "json", variant)


def metrics_artifact(dataset: str, exp: str, run: str, split: str, suffix: str, variant: str | None = None) -> Path:
    """A file next to metrics_file, e.g. suffix "per_image.parquet" for the tables of the bootstrap."""
    _check(dataset, split)
    _check_variant(dataset, variant)
    name = split if variant is None else f"{split}_{variant}"
    return out_root() / "metrics" / dataset / exp / run / f"{name}.{suffix}"


def gt_file(dataset: str, split: str, variant: str | None = None) -> Path:
    """COCO ground truth restricted to one split (all of its images, including empty ones), as evaluated
    (densepoly.eval.coco_io); variant as in annotation_file."""
    _check(dataset, split)
    _check_variant(dataset, variant)
    name = split if variant is None else f"{split}_{variant}"
    return out_root() / "gt" / dataset / f"{name}.json"
