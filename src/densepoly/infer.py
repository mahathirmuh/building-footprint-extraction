"""Predictions of an experiment on a split (scripts/predict.py; docs/EVALUATION.md §1).

A network gives per tile its channels: padded input, softmax (sigmoid on binary models), cropped back to the tile
(docs/METHOD.md §6.2 step 1), with tanh(ŝ, ĉ) behind the probabilities on FULL. The channels go through float16 whether
they are saved or not, so a prediction made from saved channels equals the one made from the network: the tuning, B5
(the network of B4) and B5′ reuse saved channels instead of running the network on CPU again.
"""

from __future__ import annotations

import multiprocessing
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn

from densepoly import paths
from densepoly.classical import b0
from densepoly.models.factory import build_model, split_output
from densepoly.postproc.polygonize import (
    InstancePolygon,
    PolygonParams,
    head_theta,
    polygonize,
    to_results,
)
from densepoly.postproc.watershed import Instances, binary_instances, three_class_instances
from densepoly.priors.cache import load_stats, quantize, read_prior, read_rgb
from densepoly.priors.mbi import brightness, mbi


def cached_mbi(dataset: str, split: str, image_id: int) -> np.ndarray:
    """MBI of one image scaled to [0, 1] (docs/METHOD.md §2.4): channel R of its prior cache divided by 255."""
    if dataset != "crowdai":
        raise NotImplementedError("B0 runs on CrowdAI only (Phase 4); WHU priors are cached per official tile")
    return read_prior(paths.prior_cache(dataset, split, image_id))[..., 0].astype(np.float32) / 255.0


@dataclass(frozen=True)
class B0Job:
    dataset: str
    split: str
    b0: b0.B0Params
    polygon: PolygonParams
    category: int


def b0_image(job: B0Job, image_id: int) -> tuple[list[dict], dict]:
    """Results of B0 on one image, and the seconds of its instance and polygon stages."""
    tic = time.perf_counter()
    instances = b0.instances(cached_mbi(job.dataset, job.split, image_id), job.b0)
    mid = time.perf_counter()
    results = to_results(polygonize(instances.instance_map, job.polygon), instances.scores, image_id, job.category)
    return results, {"instances": mid - tic, "polygons": time.perf_counter() - mid}


def _b0_chunk(args) -> list[tuple[list[dict], dict]]:
    job, image_ids = args
    return [b0_image(job, image_id) for image_id in image_ids]


@dataclass(frozen=True)
class InstanceParams:
    tau: float = 0.4  # τ
    marker_min_px: int = 8  # m_min
    instance_min_px: int = 10  # A_min

    @classmethod
    def from_config(cls, postproc) -> InstanceParams:
        return cls(float(postproc.tau), int(postproc.marker_min_px), int(postproc.instance_min_px))


@dataclass(frozen=True)
class NetworkJob:
    dataset: str
    source: str  # experiment of the checkpoint and of the saved channels
    run: str
    split: str
    instance: InstanceParams
    polygon: PolygonParams
    category: int


def network_source(cfg) -> str:
    """The experiment whose checkpoint a config uses: its source_exp (B5, B5′), or itself."""
    return str(cfg.model.source_exp or cfg.exp)


def load_network(cfg, checkpoint: Path, device: torch.device) -> tuple[nn.Module, dict]:
    """The network of a checkpoint in eval mode, and what the checkpoint says about itself. Nothing is downloaded: the
    checkpoint replaces the pretrained weights."""
    model = build_model(OmegaConf.merge(cfg, {"model": {"encoder_weights": None}}))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state.pop("model"))
    return model.to(device).eval(), state


@torch.inference_mode()
def network_channels(
    model: nn.Module, batches: Iterable[dict], device: torch.device, tile_size: int, amp: bool = False
) -> Iterator[tuple[int, np.ndarray, float]]:
    """(image id, float16 (C, H, W) channels cropped to the tile, seconds of the network per image) of every tile."""
    for batch in batches:
        tic = time.perf_counter()
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            output = model(batch["image"].to(device, non_blocking=True))
        logits, orientation = split_output(output.float())
        probabilities = torch.sigmoid(logits) if logits.shape[1] == 1 else F.softmax(logits, dim=1)
        channels = probabilities if orientation is None else torch.cat([probabilities, orientation], dim=1)
        cropped = channels[:, :, :tile_size, :tile_size].cpu().numpy().astype(np.float16)
        seconds = (time.perf_counter() - tic) / len(cropped)
        for image_id, tile in zip(batch["image_id"].tolist(), cropped, strict=True):
            yield image_id, tile, seconds


def network_instances(channels: np.ndarray, params: InstanceParams) -> Instances:
    """§6.3 on one channel (binary models), §6.2 on the three class probabilities otherwise."""
    if channels.shape[0] == 1:
        return binary_instances(channels[0], params.instance_min_px)
    return three_class_instances(*channels[:3], params.tau, params.marker_min_px, params.instance_min_px)


def network_polygons(instances: Instances, channels: np.ndarray, params: PolygonParams) -> list[InstancePolygon]:
    """Polygons of the instances; mode theta takes θ and κ from the head, weighted by 1 − P_bg (§6.4)."""
    if params.mode != "theta":
        return polygonize(instances.instance_map, params)
    theta, kappa = head_theta(instances.instance_map, channels[3], channels[4], 1.0 - channels[0])
    return polygonize(instances.instance_map, params, theta, kappa)


def network_image(job: NetworkJob, channels: np.ndarray, image_id: int) -> tuple[list[dict], dict]:
    """Results of one image from its float32 channels, and the seconds of the instance and polygon stages."""
    tic = time.perf_counter()
    instances = network_instances(channels, job.instance)
    mid = time.perf_counter()
    polygons = network_polygons(instances, channels, job.polygon)
    results = to_results(polygons, instances.scores, image_id, job.category)
    return results, {"instances": mid - tic, "polygons": time.perf_counter() - mid}


def channel_path(job: NetworkJob, image_id: int) -> Path:
    return paths.channel_file(job.dataset, job.source, job.run, job.split, image_id)


def saved_channels(job: NetworkJob, image_id: int) -> np.ndarray:
    return np.load(channel_path(job, image_id)).astype(np.float32)


def predict_network(
    job: NetworkJob,
    model: nn.Module,
    batches: Iterable[dict],
    device: torch.device,
    tile_size: int,
    save_channels: bool = False,
) -> tuple[list[dict], list[dict]]:
    """(results, timings per image) of a network on a split. The post-processing runs here, image after image: on the
    CPU of the laptop the network takes far longer anyway."""
    results, timings = [], []
    for image_id, channels, seconds in network_channels(model, batches, device, tile_size):
        if save_channels:
            target = channel_path(job, image_id)
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, channels)
        image_results, timing = network_image(job, channels.astype(np.float32), image_id)
        results += image_results
        timings.append({**timing, "network": seconds})
    return results, timings


def _saved_chunk(args) -> list[tuple[list[dict], dict]]:
    job, image_ids = args
    return [network_image(job, saved_channels(job, image_id), image_id) for image_id in image_ids]


def predict_saved(job: NetworkJob, image_ids: list[int], workers: int = 1) -> tuple[list[dict], list[dict]]:
    """(results in image order, timings per image) from the saved channels of a split."""
    parts = chunks(image_ids, max(1, min(workers, len(image_ids))) * 4)
    if workers > 1:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            done = pool.map(_saved_chunk, [(job, part) for part in parts])
    else:
        done = [_saved_chunk((job, part)) for part in parts]
    per_image = [item for part in done for item in part]
    return [r for results, _ in per_image for r in results], [timing for _, timing in per_image]


def chunks(items: list, n: int) -> list[list]:
    """n contiguous chunks of about equal size, so that concatenating their outputs keeps the order of items."""
    bounds = np.linspace(0, len(items), n + 1).astype(int)
    return [items[a:b] for a, b in zip(bounds[:-1], bounds[1:], strict=True) if b > a]


def predict_b0(job: B0Job, image_ids: list[int], workers: int = 1) -> tuple[list[dict], list[dict]]:
    """(results in image order, timings per image) of B0 on a split."""
    parts = chunks(image_ids, max(1, min(workers, len(image_ids))) * 4)
    if workers > 1:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            done = pool.map(_b0_chunk, [(job, part) for part in parts])
    else:
        done = [_b0_chunk((job, part)) for part in parts]
    per_image = [item for part in done for item in part]
    return [r for results, _ in per_image for r in results], [timing for _, timing in per_image]


def b0_seconds_from_rgb(job: B0Job, images: list[dict], mbi_scales, n: int, seed: int = 0) -> dict:
    """Seconds per image of B0 in one process from the RGB tile, MBI included (reading the file excluded), on n random
    images of the split; also checks that the MBI computed again equals the cached one."""
    q = load_stats(job.dataset)["q"][0]
    picked = np.random.default_rng(seed).choice(len(images), size=min(n, len(images)), replace=False)
    seconds, mbi_seconds, same = [], [], 0
    for i in sorted(picked.tolist()):
        image = images[i]
        rgb = read_rgb(paths.image_dir(job.dataset, job.split) / image["file_name"])
        tic = time.perf_counter()
        scaled = quantize(mbi(brightness(rgb), list(mbi_scales))[..., None], [q])[..., 0].astype(np.float32) / 255.0
        mid = time.perf_counter()
        instances = b0.instances(scaled, job.b0)
        to_results(polygonize(instances.instance_map, job.polygon), instances.scores, image["id"], job.category)
        seconds.append(time.perf_counter() - tic)
        mbi_seconds.append(mid - tic)
        same += np.array_equal(scaled, cached_mbi(job.dataset, job.split, image["id"]))
    return {
        "images": len(seconds),
        "seconds_per_image_median": float(np.median(seconds)),
        "seconds_per_image_mean": float(np.mean(seconds)),
        "mbi_seconds_per_image_median": float(np.median(mbi_seconds)),
        "processes": 1,
        "includes": "MBI from RGB, B0 instances and polygons; not reading the file",
        "mbi_equal_to_cache": f"{same}/{len(seconds)}",
    }
