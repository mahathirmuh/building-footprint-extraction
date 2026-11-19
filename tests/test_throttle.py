"""The cap on the share of time the GPU works (DENSEPOLY_GPU_DUTY)."""

import pytest
import torch

from densepoly import train
from densepoly.utils import throttle
from densepoly.utils.throttle import ENV, GpuPause, gpu_duty, refuse_throttled


def test_no_variable_means_no_cap(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert gpu_duty() is None
    monkeypatch.setenv(ENV, "  ")
    assert gpu_duty() is None
    assert not GpuPause(torch.device("cuda")).active
    refuse_throttled("anything")


def test_the_variable_is_a_fraction(monkeypatch):
    monkeypatch.setenv(ENV, "0.55")
    assert gpu_duty() == 0.55
    for wrong in ("0", "-0.2", "1.5", "65"):
        monkeypatch.setenv(ENV, wrong)
        with pytest.raises(ValueError, match=ENV):
            gpu_duty()


def test_the_sleep_leaves_the_work_at_its_share_of_the_time():
    pause = GpuPause(torch.device("cuda"), duty=0.6)
    assert pause.active
    busy = 0.3
    assert busy / (busy + pause.seconds(busy)) == pytest.approx(0.6)
    assert GpuPause(torch.device("cuda"), duty=1.0).seconds(busy) == 0.0


def test_rest_waits_for_the_card_and_then_sleeps(monkeypatch):
    calls = []
    monkeypatch.setattr(throttle.torch.cuda, "synchronize", lambda device: calls.append(("sync", device.type)))
    monkeypatch.setattr(throttle.time, "perf_counter", lambda: 10.5)
    monkeypatch.setattr(throttle.time, "sleep", lambda seconds: calls.append(("sleep", seconds)))
    GpuPause(torch.device("cuda"), duty=0.5).rest(started=10.0)
    assert calls == [("sync", "cuda"), ("sleep", pytest.approx(0.5))]


def test_nothing_sleeps_on_a_cpu(monkeypatch):
    monkeypatch.setenv(ENV, "0.5")
    monkeypatch.setattr(throttle.time, "sleep", lambda seconds: pytest.fail("slept on a CPU"))
    pause = GpuPause(torch.device("cpu"))
    assert not pause.active
    pause.rest(started=0.0)


def test_a_throttled_run_is_not_a_speed_measurement(monkeypatch):
    monkeypatch.setenv(ENV, "0.5")
    refuse_throttled("a benchmark on the CPU")  # the tests see no GPU: nothing is capped there
    monkeypatch.setattr(throttle.torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="measures speed"):
        refuse_throttled("the benchmark")
    with pytest.raises(RuntimeError, match="measures speed"):
        train.benchmark(cfg=None, iterations=1)
