"""A cap on the share of time the GPU works, for a card that must not be loaded fully (asked for on 19 Nov 2026, after
the card of the lab server fell off the bus under a training run).

The environment variable ``DENSEPOLY_GPU_DUTY`` holds a fraction in (0, 1]: after every piece of GPU work (an optimizer
step, a validation batch, a batch of predictions) the process sleeps long enough for the work to fill at most that
fraction of the wall time. It is an operating setting like the paths, not part of an experiment: no config key, no
change of any result. Unset, nothing sleeps. A throttled run does not measure speed, so the benchmarks refuse it.
"""

from __future__ import annotations

import os
import time

import torch

ENV = "DENSEPOLY_GPU_DUTY"


def gpu_duty() -> float | None:
    """The fraction asked for, or None when the variable is unset or empty."""
    value = os.environ.get(ENV, "").strip()
    if not value:
        return None
    duty = float(value)
    if not 0.0 < duty <= 1.0:
        raise ValueError(f"{ENV} must be in (0, 1], got {value!r}")
    return duty


def refuse_throttled(what: str) -> None:
    """Speed measured between sleeps is not the speed of the model, and a benchmark without them loads the card fully:
    with a cap below 1 a benchmark on the GPU does not run at all. On a CPU the cap does nothing, nor does this."""
    duty = gpu_duty()
    if duty is not None and duty < 1.0 and torch.cuda.is_available():
        raise RuntimeError(f"{what} measures speed and cannot run on the GPU with {ENV}={duty}")


class GpuPause:
    """Sleeps after GPU work so that it fills at most `duty` of the wall time; inactive on a CPU or without a duty."""

    def __init__(self, device: torch.device, duty: float | None = None):
        self.device = device
        self.duty = gpu_duty() if duty is None else duty
        self.active = self.duty is not None and self.duty < 1.0 and device.type == "cuda"

    def seconds(self, busy: float) -> float:
        """The sleep that follows `busy` seconds of work."""
        return busy * (1.0 / self.duty - 1.0) if self.active else 0.0

    def rest(self, started: float) -> None:
        """Call after work that began at `started` (time.perf_counter()): waits for the queued kernels, then sleeps."""
        if not self.active:
            return
        torch.cuda.synchronize(self.device)
        time.sleep(self.seconds(time.perf_counter() - started))
