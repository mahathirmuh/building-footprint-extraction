"""PyTorch dataset of one split (docs/METHOD.md §2.4, §3.5; docs/EXPERIMENTS.md §3).

Three modes:
- train: the augmented tile with all its targets, derived from the transformed instance map (transforms.make_sample);
- val: the tile as it is with its classes, for the foreground IoU and the contact F1 of every epoch;
- predict: the tile as it is with its image id.

The augmentation of a sample is a pure function of (seed, epoch, index) and no random state lives in the workers, so a
batch does not depend on the number of workers, and a run resumed in the middle of an epoch draws what it would have
drawn. EpochSampler hands (epoch, index) to the dataset, because persistent workers keep their own copy of the dataset
and would never see a new epoch set on it.

A CrowdAI sample is one tile. A WHU sample is one patch cut from its official tile at the offset in patches.csv
(docs/DATA.md §3.2), with the instance map of the patch and the priors cut from the cached map of that tile.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from densepoly import paths
from densepoly.data.transforms import make_sample, pad
from densepoly.labels.contact import class_map
from densepoly.priors.cache import CHANNELS, load_stats, normalize, read_prior, read_rgb, units
from densepoly.utils.io import read_lines, read_png

MODES = ("train", "val", "predict")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass(frozen=True)
class TileRef:
    """Where the pixels of one sample are."""

    key: str  # image id: names the instance map and identifies the predictions
    image: Path  # the file holding the pixels
    prior_key: str  # names the prior map
    offset: tuple[int, int] | None = None  # (x, y) of the sample inside the file; None when the file is the sample


def tile_refs(dataset: str, split: str, splits_dir: Path | None = None) -> list[TileRef]:
    """The samples of a split, in the order of its list."""
    if dataset == "crowdai":
        return [TileRef(unit.key, unit.image, unit.key) for unit in units(dataset, split, splits_dir)]
    tiles = {unit.key: unit.image for unit in units(dataset, split, splits_dir)}
    with paths.patch_index(dataset, split).open(encoding="utf-8") as f:
        patches = {row["file_name"]: row for row in csv.DictReader(f)}
    rows = [patches[name] for name in read_lines(paths.split_list(dataset, split, splits_dir))]
    return [TileRef(row["image_id"], tiles[row["tile"]], row["tile"], (int(row["x"]), int(row["y"]))) for row in rows]


class EpochSampler(Sampler):
    """The samples of one epoch as (epoch, index), shuffled by (seed, epoch). skip drops the first samples of the order,
    which a resumed run has already seen."""

    def __init__(self, length: int, seed: int):
        self.length, self.seed = length, seed
        self.epoch, self.skip = 0, 0

    def set_epoch(self, epoch: int, skip: int = 0) -> None:
        self.epoch, self.skip = epoch, skip

    def __iter__(self):
        order = np.random.default_rng([self.seed, self.epoch]).permutation(self.length)
        return iter([(self.epoch, int(index)) for index in order[self.skip :]])

    def __len__(self) -> int:
        return self.length - self.skip


class TileDataset(Dataset):
    def __init__(self, cfg, split: str, mode: str, seed: int = 0, splits_dir: Path | None = None):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        self.cfg, self.split, self.mode, self.seed = cfg, split, mode, seed
        self.dataset = str(cfg.data.name)
        self.units = tile_refs(self.dataset, split, splits_dir)
        self.prior_channels = [CHANNELS.index(name) for name in cfg.input.priors]
        if self.prior_channels:
            stats = load_stats(self.dataset)
            self.prior_stats = {key: [stats[key][c] for c in self.prior_channels] for key in ("mean", "std")}

    def __len__(self) -> int:
        return len(self.units)

    def cut(self, array: np.ndarray, ref: TileRef) -> np.ndarray:
        """The part of a file's array that is this sample: all of it for a whole tile, else the patch at its offset."""
        if ref.offset is None:
            return array
        x, y = ref.offset
        size = int(self.cfg.data.tile_size)
        patch = array[y : y + size, x : x + size]
        if patch.shape[:2] != (size, size):
            raise ValueError(f"patch at {ref.offset} of size {size} lies outside {ref.image.name}, {array.shape[:2]}")
        return patch

    def read_priors(self, ref: TileRef) -> np.ndarray | None:
        """(H, W, k) uint8 prior channels of the experiment, or None for RGB-only models."""
        if not self.prior_channels:
            return None
        prior = read_prior(paths.prior_cache(self.dataset, self.split, ref.prior_key))
        return self.cut(prior, ref)[..., self.prior_channels]

    def model_input(self, image: np.ndarray, priors: np.ndarray | None) -> torch.Tensor:
        """(C, H, W) float32: RGB with the ImageNet statistics, then the priors with those of the train split."""
        channels = (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        if priors is not None:
            channels = np.dstack([channels, normalize(priors, self.prior_stats)])
        return torch.from_numpy(channels.transpose(2, 0, 1).copy())

    def plain_input(self, image: np.ndarray, priors: np.ndarray | None) -> torch.Tensor:
        size = int(self.cfg.data.pad_to)
        return self.model_input(pad(image, size), None if priors is None else pad(priors, size))

    def __getitem__(self, item) -> dict:
        epoch, index = item if isinstance(item, tuple) else (0, item)
        ref = self.units[index]
        image, priors = self.cut(read_rgb(ref.image), ref), self.read_priors(ref)
        if self.mode == "predict":
            return {"image": self.plain_input(image, priors), "image_id": int(ref.key)}
        instance_map = read_png(paths.instance_cache(self.dataset, self.split, ref.key))
        if self.mode == "val":
            classes = class_map(instance_map, int(self.cfg.labels.contact_radius_px))
            padded = pad(classes, int(self.cfg.data.pad_to), int(self.cfg.data.ignore_index))
            return {"image": self.plain_input(image, priors), "classes": torch.from_numpy(padded.astype(np.int64))}
        sample = make_sample(image, priors, instance_map, self.cfg, np.random.default_rng([self.seed, epoch, index]))
        targets = sample.targets
        return {
            "image": self.model_input(sample.image, sample.priors),
            "classes": torch.from_numpy(targets.classes.astype(np.int64)),
            "weight": torch.from_numpy(targets.weight),
            "sin4": torch.from_numpy(targets.sin4),
            "cos4": torch.from_numpy(targets.cos4),
            "valid": torch.from_numpy(targets.valid),
        }
