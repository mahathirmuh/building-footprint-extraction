"""Cost of a model (docs/EXPERIMENTS.md §6): number of parameters, speed of the network at batch 1, and time of the
post-processing per image on the CPU. scripts/benchmark.py measures them on a trained run."""

from __future__ import annotations

import platform
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from torch import nn

from densepoly import infer

WARMUP, ITERATIONS = 20, 200


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


@torch.inference_mode()
def network_speed(
    model: nn.Module,
    shape: tuple[int, int, int, int],
    device: torch.device,
    amp: bool = True,
    warmup: int = WARMUP,
    iterations: int = ITERATIONS,
) -> dict:
    """Forward passes per second on a random input of this shape: `warmup` passes that are not timed, then `iterations`
    timed ones. On a GPU the clock waits for the device before and after, and AMP runs in float16 as in training."""
    model = model.to(device).eval()
    batch = torch.randn(shape, device=device)
    on_gpu = device.type == "cuda"

    def forward() -> None:
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp and on_gpu):
            model(batch)

    for _ in range(warmup):
        forward()
    if on_gpu:
        torch.cuda.synchronize(device)
    tic = time.perf_counter()
    for _ in range(iterations):
        forward()
    if on_gpu:
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - tic
    return {
        "fps": iterations * shape[0] / seconds,
        "network_ms": 1000 * seconds / iterations,
        "warmup": warmup,
        "iterations": iterations,
        "amp": bool(amp and on_gpu),
        "input": list(shape),
    }


def postproc_time(job: infer.NetworkJob, channels: Iterable[tuple[int, np.ndarray]]) -> dict:
    """Mean milliseconds per image of the instance and polygon stages, in this process, over (image id, channels)."""
    stages = [infer.network_image(job, tile.astype(np.float32), image_id)[1] for image_id, tile in channels]
    if not stages:
        raise ValueError("no image to time")
    instances = 1000 * float(np.mean([s["instances"] for s in stages]))
    polygons = 1000 * float(np.mean([s["polygons"] for s in stages]))
    return {"images": len(stages), "instances_ms": instances, "polygons_ms": polygons, "total_ms": instances + polygons}


def cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()
