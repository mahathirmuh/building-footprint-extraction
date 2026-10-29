"""Figures. Phase 2 QA (PLAN.md):

    python scripts/figures.py --qa-labels --data crowdai    # 12 val_tune tiles: RGB, classes, θ, w, MBI, gap map
    python scripts/figures.py --qa-gap-mode --data crowdai  # 10 val_tune tiles: gap map in max vs contrast mode

Tiles come from the frozen density strata of val_tune (seed 0), so that sparse, dense and touching tiles all appear;
the test split is never shown. Priors are computed on the fly with the dataset's settings (WHU on the whole official
tile, then cropped); their panels are scaled by the 99.5th percentile over the shown tiles, for display only.
Output: $DENSEPOLY_OUT/figures/qa/<dataset>_labels.png and <dataset>_gap_mode.png.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import hsv_to_rgb  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from densepoly import paths  # noqa: E402
from densepoly.config import load_config  # noqa: E402
from densepoly.labels.boundary_weight import boundary_mask  # noqa: E402
from densepoly.labels.instances import load_instances  # noqa: E402
from densepoly.labels.targets import targets_from_config  # noqa: E402
from densepoly.priors import cache  # noqa: E402
from densepoly.priors.gap import gap_map  # noqa: E402
from densepoly.priors.mbi import brightness  # noqa: E402
from densepoly.utils.io import read_json  # noqa: E402

SPLIT = "val_tune"
CLASS_COLOURS = np.array([[0, 0, 0], [200, 200, 200], [230, 40, 40]], np.uint8)


def pick_tiles(dataset: str, plan: dict[str, int], splits_dir: Path | None, seed: int = 0) -> list[tuple[int, str]]:
    """(image id, stratum) drawn without repetition from the val_tune strata; plan maps a stratum to a count."""
    strata = read_json(paths.density_file(dataset, SPLIT, splits_dir))
    groups = {**strata["main"]["strata"], "bersentuhan": strata["secondary"]["strata"]["bersentuhan"]}
    rng, taken, picked = np.random.default_rng(seed), set(), []
    for stratum, count in plan.items():
        pool = [i for i in groups[stratum] if i not in taken]
        for i in rng.permutation(len(pool))[:count]:
            taken.add(pool[i])
            picked.append((pool[i], stratum))
    return picked


class TileSource:
    """RGB, instance map and priors of val_tune images: a CrowdAI tile, or a WHU patch cut from its official tile."""

    def __init__(self, dataset: str):
        self.dataset = dataset
        self.cfg = load_config(dataset, "full_p2")
        self.priors = OmegaConf.to_container(self.cfg.priors)
        index = paths.instance_dir(dataset, SPLIT) / "index.csv"
        with index.open(encoding="utf-8") as f:
            self.file_names = {int(row["image_id"]): row["file_name"] for row in csv.DictReader(f)}
        self.patches, self.tiles, self.tile_priors = {}, {}, {}
        if dataset == "whu":
            with paths.patch_index(dataset, SPLIT).open(encoding="utf-8") as f:
                self.patches = {int(row["image_id"]): row for row in csv.DictReader(f)}

    def _window(self, image_id: int):
        row = self.patches[image_id]
        y, x = int(row["y"]), int(row["x"])
        return row["tile"], (slice(y, y + 224), slice(x, x + 224))

    def _whu_tile(self, stem: str) -> np.ndarray:
        if stem not in self.tiles:
            folder = paths.dataset_dir("whu") / "raw_0.2" / paths.source_folder("whu", SPLIT) / "image"
            self.tiles[stem] = cache.read_rgb(folder / f"{stem}.tif")
        return self.tiles[stem]

    def rgb(self, image_id: int) -> np.ndarray:
        if self.dataset == "crowdai":
            return cache.read_rgb(paths.image_dir("crowdai", SPLIT) / self.file_names[image_id])
        stem, window = self._window(image_id)
        return self._whu_tile(stem)[window]

    def instances(self, image_id: int) -> np.ndarray:
        return load_instances(self.dataset, SPLIT, image_id)

    def brightness(self, image_id: int) -> tuple[np.ndarray, tuple]:
        """Brightness of the unit the priors are computed on, and the window of the image inside it."""
        if self.dataset == "crowdai":
            return brightness(self.rgb(image_id)), (slice(None), slice(None))
        stem, window = self._window(image_id)
        return brightness(self._whu_tile(stem)), window

    def prior_maps(self, image_id: int) -> np.ndarray:
        if self.dataset == "crowdai":
            return cache.compute_priors(self.rgb(image_id), self.priors)
        stem, window = self._window(image_id)
        if stem not in self.tile_priors:  # ±13 s per official tile: computed once for all its patches
            self.tile_priors[stem] = cache.compute_priors(self._whu_tile(stem), self.priors)
        return self.tile_priors[stem][window]


def theta_rgb(instance_map: np.ndarray, targets) -> np.ndarray:
    """θ = atan2(sin 4θ, cos 4θ) / 4 as hue (0° to 90°), buildings without a θ target in grey, background black."""
    theta = (np.arctan2(targets.sin4, targets.cos4) / 4) % (np.pi / 2)
    hue = np.where(targets.valid, theta / (np.pi / 2), 0.0).astype(np.float32)
    rgb = hsv_to_rgb(np.dstack([hue, np.ones_like(hue), targets.valid.astype(np.float32)]))
    rgb[(instance_map > 0) & ~targets.valid] = 0.5
    return rgb


def outline(image: np.ndarray, instance_map: np.ndarray) -> np.ndarray:
    shown = image.copy()
    shown[boundary_mask(instance_map) & (instance_map > 0)] = (255, 40, 40)
    return shown


def save(fig, dataset: str, name: str) -> Path:
    target = paths.out_root() / "figures" / "qa" / f"{dataset}_{name}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=80)
    plt.close(fig)
    return target


def qa_labels(dataset: str, splits_dir: Path | None = None, seed: int = 0) -> Path:
    source = TileSource(dataset)
    picked = pick_tiles(dataset, {"jarang": 3, "sedang": 3, "padat": 3, "bersentuhan": 3}, splits_dir, seed)
    rows = []
    for image_id, stratum in picked:
        instance_map = source.instances(image_id)
        rows.append((image_id, stratum, source.rgb(image_id), instance_map, source.prior_maps(image_id)))
    top = np.percentile(np.concatenate([r[4].reshape(-1, 2) for r in rows]), 99.5, axis=0)
    alpha = float(source.cfg.labels.edge_weight_alpha)
    titles = ["RGB", "kelas (abu bangunan, merah kontak)", "θ (warna = sudut)", "bobot tepi w", "MBI", "peta celah"]
    fig, axes = plt.subplots(len(rows), 6, figsize=(15, 2.6 * len(rows)), squeeze=False)
    for row_axes, (image_id, stratum, rgb, instance_map, prior) in zip(axes, rows, strict=True):
        targets = targets_from_config(instance_map, source.cfg.labels)
        panels = [
            (rgb, {}),
            (CLASS_COLOURS[targets.classes], {}),
            (theta_rgb(instance_map, targets), {}),
            (targets.weight, {"cmap": "viridis", "vmin": 1.0, "vmax": 1.0 + alpha}),
            (prior[..., 0], {"cmap": "magma", "vmin": 0.0, "vmax": top[0]}),
            (prior[..., 1], {"cmap": "magma", "vmin": 0.0, "vmax": top[1]}),
        ]
        for ax, (array, style) in zip(row_axes, panels, strict=True):
            ax.imshow(array, interpolation="nearest", **style)
            ax.set_xticks([])
            ax.set_yticks([])
        row_axes[0].set_ylabel(f"{image_id}\n{stratum}", fontsize=8)
    for ax, title in zip(axes[0], titles, strict=True):
        ax.set_title(title, fontsize=9)
    fig.suptitle(f"{dataset} val_tune: label dan prior (r = {source.cfg.labels.contact_radius_px} px)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))  # leave room for the suptitle above the column titles
    return save(fig, dataset, "labels")


def qa_gap_mode(dataset: str, splits_dir: Path | None = None, seed: int = 0) -> Path:
    source = TileSource(dataset)
    length = int(source.priors["gap_se_length_px"])
    rows = []
    for image_id, stratum in pick_tiles(dataset, {"bersentuhan": 6, "padat": 4}, splits_dir, seed):
        b, window = source.brightness(image_id)
        maps = [gap_map(b, length, mode)[window] for mode in ("max", "contrast")]
        rows.append((image_id, stratum, source.rgb(image_id), source.instances(image_id), maps))
    tops = [np.percentile(np.concatenate([r[4][k].ravel() for r in rows]), 99.5) for k in range(2)]
    titles = ["RGB", "RGB + batas GT", f"celah max (L_g = {length})", f"celah contrast (L_g = {length})"]
    fig, axes = plt.subplots(len(rows), 4, figsize=(11, 2.8 * len(rows)), squeeze=False)
    for row_axes, (image_id, stratum, rgb, instance_map, maps) in zip(axes, rows, strict=True):
        panels = [(rgb, {}), (outline(rgb, instance_map), {})]
        panels += [(m, {"cmap": "magma", "vmin": 0.0, "vmax": top}) for m, top in zip(maps, tops, strict=True)]
        for ax, (array, style) in zip(row_axes, panels, strict=True):
            ax.imshow(array, interpolation="nearest", **style)
            ax.set_xticks([])
            ax.set_yticks([])
        row_axes[0].set_ylabel(f"{image_id}\n{stratum}", fontsize=8)
    for ax, title in zip(axes[0], titles, strict=True):
        ax.set_title(title, fontsize=9)
    fig.suptitle(f"{dataset} val_tune: mode peta celah", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))  # leave room for the suptitle above the column titles
    return save(fig, dataset, "gap_mode")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", choices=paths.DATASETS, required=True)
    parser.add_argument("--qa-labels", action="store_true")
    parser.add_argument("--qa-gap-mode", action="store_true")
    parser.add_argument("--splits-dir", type=Path, help="folder of the density strata (default: repo splits/)")
    args = parser.parse_args(argv)
    if not (args.qa_labels or args.qa_gap_mode):
        parser.error("nothing to do: give --qa-labels and/or --qa-gap-mode")
    if args.qa_labels:
        print("saved", qa_labels(args.data, args.splits_dir))
    if args.qa_gap_mode:
        print("saved", qa_gap_mode(args.data, args.splits_dir))


if __name__ == "__main__":
    main()
