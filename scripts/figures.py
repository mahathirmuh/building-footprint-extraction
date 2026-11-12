"""Figures. Phase 2 QA (PLAN.md) and the result figures of Phase 8:

    python scripts/figures.py --qa-labels --data crowdai    # 12 val_tune tiles: RGB, classes, θ, w, MBI, gap map
    python scripts/figures.py --qa-gap-mode --data crowdai  # 10 val_tune tiles: gap map in max vs contrast mode
    python scripts/figures.py --results --data crowdai      # density analysis, from the tables of scripts/tables.py
    python scripts/figures.py --qualitative --data crowdai  # real test tiles: ground truth and the polygons of each model

--results draws no number of its own: it reads $DENSEPOLY_OUT/tables/<dataset>/n2.json (effect of each comparison per
density stratum, with its 95% CI) and strata.json (metrics per stratum), and writes <dataset>_n2 and <dataset>_strata as
PNG and PDF to $DENSEPOLY_OUT/figures/results/. --lang picks the language of the labels (en for the paper, id for the
slides).

--qualitative draws the saved test predictions of seed 0 (outputs/preds/) over the tiles: QUALITATIVE_DENSE tiles drawn at
random (seed 0) from the dense stratum and, as the failure case, the tile with touching buildings on which the FULL final
merges the most buildings (from its per-image metrics). It writes <dataset>_qualitative as PNG and PDF.

The QA figures:

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
from densepoly.eval.tables import LABELS  # noqa: E402
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


# --- result figures (Phase 8): colours of a validated palette, text always in ink, one y axis per panel

SERIES_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")  # at most three series per panel
SERIES_MARKERS = ("o", "s", "D")  # a second cue next to the colour, for print
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DENSITY, TOUCHING = ("jarang", "sedang", "padat"), ("bersentuhan", "tidak_bersentuhan")
STRATA_SERIES = {"crowdai": ("b3", "b4", "full_p0"), "whu": ("b5p", "full_p0")}
STRATA_PANELS = ("AP", "merge_rate", "PoLiS_paired")
SCALE = {"AP": 100, "merge_rate": 100, "split_rate": 100, "CIoU": 100}  # shown in percent
WORDS = {
    "en": {
        "jarang": "sparse", "sedang": "medium", "padat": "dense",
        "bersentuhan": "touching", "tidak_bersentuhan": "not touching",
        "AP": "AP (%)", "merge_rate": "merge rate (%)", "PoLiS_paired": "PoLiS, paired (px)",
        "MTA_inst": "MTA per instance (°)", "delta": "Δ", "points": "points",
        "density": "tiles by number of buildings", "tiles": "tiles", "decimal": ".",
    },
    "id": {
        "jarang": "jarang", "sedang": "sedang", "padat": "padat",
        "bersentuhan": "bersentuhan", "tidak_bersentuhan": "tidak bersentuhan",
        "AP": "AP (%)", "merge_rate": "merge rate (%)", "PoLiS_paired": "PoLiS berpasangan (px)",
        "MTA_inst": "MTA per instance (°)", "delta": "Δ", "points": "poin",
        "density": "tile menurut jumlah bangunan", "tiles": "tile", "decimal": ",",
    },
}  # fmt: skip


def read_table(dataset: str, name: str) -> dict:
    file = paths.out_root() / "tables" / dataset / f"{name}.json"
    if not file.exists():
        raise FileNotFoundError(f"{file} does not exist: run scripts/tables.py --data {dataset} --table {name} first")
    return read_json(file)


DIGITS = {"PoLiS_paired": 3}


def number(value: float, words: dict, digits: int = 2, signed: bool = False) -> str:
    return f"{value:{'+' if signed else ''}.{digits}f}".replace(".", words["decimal"]).replace("-", "−")


def spread(positions: list[float], gap: float) -> list[float]:
    """The positions moved apart, in their order, so that neighbours are at least gap away; their mean is kept."""
    order = sorted(range(len(positions)), key=positions.__getitem__)
    placed = []
    for i in order:
        placed.append(max(positions[i], placed[-1] + gap) if placed else positions[i])
    shift = sum(positions[i] for i in order) / len(order) - sum(placed) / len(placed) if placed else 0.0
    result = [0.0] * len(positions)
    for i, value in zip(order, placed, strict=True):
        result[i] = value + shift
    return result


def style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save_result(fig, dataset: str, name: str) -> Path:
    target = paths.out_root() / "figures" / "results" / f"{dataset}_{name}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=200, facecolor="white")
    fig.savefig(target.with_suffix(".pdf"), facecolor="white", metadata={"CreationDate": None})  # same bytes each time
    plt.close(fig)
    return target


def results_n2(dataset: str, lang: str = "en") -> Path:
    """Δ of every comparison of the n2 table in each stratum, with its 95% CI: one panel per comparison and metric."""
    words, table = WORDS[lang], read_table(dataset, "n2")
    panels = [(name, spec, metric) for name, spec in table["comparisons"].items() for metric in spec["metrics"]]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6), squeeze=False)
    for ax, (name, spec, metric) in zip(axes[0], panels, strict=True):
        groups = [g for g in (*DENSITY, *TOUCHING) if g in spec["per_stratum"]]
        scale = SCALE.get(metric, 1)
        first = [g for g in groups if g in DENSITY]  # then, after a gap, the second grouping with room for its names
        xs = [first.index(g) if g in first else len(first) + 0.6 + 1.4 * TOUCHING.index(g) for g in groups]
        deltas = np.array([spec["per_stratum"][g][metric]["delta"] for g in groups], float) * scale
        cis = np.array([spec["per_stratum"][g][metric]["ci95"] or [np.nan, np.nan] for g in groups], float) * scale
        ax.axhline(0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.errorbar(
            xs, deltas, yerr=[deltas - cis[:, 0], cis[:, 1] - deltas], fmt="o", color=SERIES_COLOURS[0],
            markersize=7, markeredgecolor="white", markeredgewidth=1.5, elinewidth=1.5, capsize=3,
        )  # fmt: skip
        values = [number(delta, words, DIGITS.get(metric, 2), signed=True) for delta in deltas]  # under the tick,
        names = [words[g].replace(" ", "\n") if g in TOUCHING else words[g] for g in groups]
        ticks = [f"{name}\n{value}" for name, value in zip(names, values, strict=True)]  # clear of the intervals
        ax.set_xticks(xs, ticks, fontsize=8)
        ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
        unit = f"{words['delta']} {words[metric]}".replace("(%)", f"({words['points']})")
        ax.set_ylabel(unit, color=MUTED, fontsize=9)
        left, right = (LABELS[exp] for exp in name.split("->"))
        ax.set_title(f"{spec['name']}: {left} → {right}", color=INK, fontsize=10, loc="left")
        style(ax)
    fig.tight_layout()
    return save_result(fig, dataset, "n2" if lang == "en" else f"n2_{lang}")


def results_strata(dataset: str, lang: str = "en") -> Path:
    """Metrics per density stratum of up to three experiments, mean over the seeds with ± SD bars; one panel per metric."""
    words, table = WORDS[lang], read_table(dataset, "strata")
    series = [exp for exp in STRATA_SERIES[dataset] if exp in table["rows"]]
    fig, axes = plt.subplots(1, len(STRATA_PANELS), figsize=(3.8 * len(STRATA_PANELS), 3.7), squeeze=False)
    xs = np.arange(len(DENSITY))
    handles = {}
    for ax, metric in zip(axes[0], STRATA_PANELS, strict=True):
        scale = SCALE.get(metric, 1)
        ends = {}
        for k, exp in enumerate(series):
            summaries = [table["rows"][exp]["strata"][g][metric]["summary"] for g in DENSITY]
            if any(s is None for s in summaries):
                continue
            means = np.array([s["mean"] for s in summaries]) * scale
            sds = np.array([s["sd"] or 0.0 for s in summaries]) * scale
            handles[exp] = ax.errorbar(
                xs, means, yerr=sds, color=SERIES_COLOURS[k], marker=SERIES_MARKERS[k], markersize=7,
                markeredgecolor="white", markeredgewidth=1.5, linewidth=2, elinewidth=1.2, capsize=2,
            )  # fmt: skip
            ends[exp] = float(means[-1])
        low, high = ax.get_ylim()
        labels_at = spread(list(ends.values()), 0.07 * (high - low))  # names at the line ends, kept apart
        for exp, y in zip(ends, labels_at, strict=True):
            ax.annotate(LABELS[exp], (xs[-1], y), xytext=(10, 0), textcoords="offset points", va="center",
                        fontsize=8, color=INK)  # fmt: skip
        sizes = [table["sizes"][g]["n_images"] for g in DENSITY]
        ticks = [f"{words[g]}\n({n} {words['tiles']})" for g, n in zip(DENSITY, sizes, strict=True)]
        ax.set_xticks(xs, ticks, fontsize=8)
        ax.set_xlim(-0.35, len(DENSITY) - 1 + 0.75)
        ax.set_title(words[metric], color=INK, fontsize=10, loc="left")
        style(ax)
    fig.legend(
        handles.values(), [LABELS[exp] for exp in handles], loc="upper center", ncol=len(handles), frameon=False,
        fontsize=9, labelcolor=INK,
    )  # fmt: skip
    fig.supxlabel(words["density"], color=MUTED, fontsize=9, y=0.04)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    return save_result(fig, dataset, "strata" if lang == "en" else f"strata_{lang}")


# --- qualitative figure: what each model draws on the same real tiles

QUALITATIVE_MODELS = {"crowdai": ("b4", "full_p0", "full_y", "b3m"), "whu": ("b5p", "full_p0")}
QUALITATIVE_DENSE = 3
GT_COLOUR, PREDICTION_COLOUR = "#39d353", "#ffd60a"
CAPTIONS = {
    "en": {"gt": "ground truth", "dense": "dense tile", "failure": "failure case (merged buildings)"},
    "id": {"gt": "ground truth", "dense": "tile padat", "failure": "contoh kegagalan (rumah tergabung)"},
}


def polygons_by_image(items: list[dict], image_ids: set[int]) -> dict[int, list[np.ndarray]]:
    """{image id: [(n, 2) rings]} of the COCO annotations or results of these images (first ring of each)."""
    rings: dict[int, list[np.ndarray]] = {image_id: [] for image_id in image_ids}
    for item in items:
        if item["image_id"] in rings and item.get("segmentation"):
            rings[item["image_id"]].append(np.reshape(item["segmentation"][0], (-1, 2)))
    return rings


def pick_qualitative(dense: list[int], touching: list[int], merged: dict[int, int], n_dense: int, seed: int = 0) -> list:
    """[(image id, kind)]: n_dense tiles drawn from the dense stratum, then the touching tile with the most merged
    buildings (ties: the smallest id) as the failure case, when there is one."""
    rng = np.random.default_rng(seed)
    picked = [(int(i), "dense") for i in sorted(rng.choice(sorted(dense), min(n_dense, len(dense)), replace=False).tolist())]
    taken = {image_id for image_id, _ in picked}
    candidates = [i for i in touching if i not in taken and merged.get(i, 0) > 0]
    if candidates:
        picked.append((min(candidates, key=lambda i: (-merged[i], i)), "failure"))
    return picked


def draw_rings(ax, image: np.ndarray, rings: list[np.ndarray], colour: str) -> None:
    ax.imshow(image)
    for ring in rings:
        closed = np.vstack([ring, ring[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color="black", linewidth=2.2, alpha=0.55)  # a dark edge under the colour
        ax.plot(closed[:, 0], closed[:, 1], color=colour, linewidth=1.2)
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_xticks([])
    ax.set_yticks([])


def qualitative(dataset: str, lang: str = "en", splits_dir: Path | None = None) -> Path:
    import pandas as pd

    from densepoly.data.dataset import TileDataset, tile_refs

    split, run = "test", paths.run_name(0)
    models = [exp for exp in QUALITATIVE_MODELS[dataset] if paths.pred_file(dataset, exp, run, split).exists()]
    density = read_json(paths.density_file(dataset, split, splits_dir))
    table = pd.read_parquet(paths.metrics_artifact(dataset, "full_p0", run, split, "per_image.parquet"))
    merged = dict(zip(table["image_id"].astype(int), table["n_merged"].astype(int), strict=True))
    picked = pick_qualitative(
        density["main"]["strata"]["padat"], density["secondary"]["strata"]["bersentuhan"], merged, QUALITATIVE_DENSE
    )
    image_ids = {image_id for image_id, _ in picked}
    refs = {int(ref.key): ref for ref in tile_refs(dataset, split, splits_dir) if int(ref.key) in image_ids}
    tiles = TileDataset(load_config(dataset, "full_p0"), split, "predict", splits_dir=splits_dir)
    images = {i: tiles.cut(cache.read_rgb(refs[i].image), refs[i]) for i in image_ids}
    columns = [("gt", polygons_by_image(read_json(paths.gt_file(dataset, split))["annotations"], image_ids))]
    columns += [(exp, polygons_by_image(read_json(paths.pred_file(dataset, exp, run, split)), image_ids)) for exp in models]
    words = CAPTIONS[lang]
    fig, axes = plt.subplots(len(picked), len(columns), figsize=(2.6 * len(columns), 2.7 * len(picked)), squeeze=False)
    for row, (image_id, kind) in enumerate(picked):
        for col, (name, rings) in enumerate(columns):
            ax = axes[row][col]
            draw_rings(ax, images[image_id], rings[image_id], GT_COLOUR if name == "gt" else PREDICTION_COLOUR)
            if row == 0:
                ax.set_title(words["gt"] if name == "gt" else LABELS[name], color=INK, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{words[kind]}\n#{image_id}", color=MUTED, fontsize=8)
            ax.text(0.03, 0.04, str(len(rings[image_id])), transform=ax.transAxes, fontsize=8, color="white",
                    bbox={"facecolor": "black", "alpha": 0.6, "pad": 1.5, "edgecolor": "none"})  # fmt: skip
    fig.tight_layout(pad=0.4)
    return save_result(fig, dataset, "qualitative" if lang == "en" else f"qualitative_{lang}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", choices=paths.DATASETS, required=True)
    parser.add_argument("--qa-labels", action="store_true")
    parser.add_argument("--qa-gap-mode", action="store_true")
    parser.add_argument("--results", action="store_true", help="the density figures, from the written tables")
    parser.add_argument("--qualitative", action="store_true", help="test tiles with the polygons of each model")
    parser.add_argument("--lang", choices=sorted(WORDS), default="en", help="language of the labels of the result figures")
    parser.add_argument("--splits-dir", type=Path, help="folder of the density strata (default: repo splits/)")
    args = parser.parse_args(argv)
    if not (args.qa_labels or args.qa_gap_mode or args.results or args.qualitative):
        parser.error("nothing to do: give --qa-labels, --qa-gap-mode, --results and/or --qualitative")
    if args.qualitative:
        print("saved", qualitative(args.data, args.lang, args.splits_dir))
    if args.results:
        print("saved", results_n2(args.data, args.lang))
        print("saved", results_strata(args.data, args.lang))
    if args.qa_labels:
        print("saved", qa_labels(args.data, args.splits_dir))
    if args.qa_gap_mode:
        print("saved", qa_gap_mode(args.data, args.splits_dir))


if __name__ == "__main__":
    main()
