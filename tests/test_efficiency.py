"""Cost measurements (densepoly.eval.efficiency) on tiny models and synthetic channels."""

import numpy as np
import pytest
import torch
from torch import nn

from densepoly import infer
from densepoly.eval import efficiency
from densepoly.postproc.polygonize import PolygonParams


class Counting(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.conv(x)


def test_parameters_are_counted_with_their_biases():
    assert efficiency.count_parameters(nn.Conv2d(3, 4, 3)) == 3 * 4 * 9 + 4
    assert efficiency.count_parameters(nn.Sequential(nn.Linear(5, 2), nn.Linear(2, 1, bias=False))) == 12 + 2


def test_network_speed_times_the_passes_after_the_warmup():
    model = Counting()
    speed = efficiency.network_speed(model, (1, 3, 16, 16), torch.device("cpu"), amp=True, warmup=3, iterations=7)
    assert model.calls == 10 and not model.training
    assert speed["fps"] > 0 and speed["network_ms"] == pytest.approx(1000 / speed["fps"])
    assert speed["amp"] is False  # float16 autocast is for the GPU only
    assert (speed["warmup"], speed["iterations"], speed["input"]) == (3, 7, [1, 3, 16, 16])


def test_speed_counts_every_tile_of_a_larger_batch():
    speed = efficiency.network_speed(Counting(), (4, 3, 16, 16), torch.device("cpu"), warmup=1, iterations=5)
    assert speed["fps"] == pytest.approx(4 * 1000 / speed["network_ms"])


def test_postproc_time_is_the_mean_of_both_stages():
    probabilities = np.zeros((3, 40, 40), np.float16)
    probabilities[0] = 1.0
    probabilities[:, 10:30, 10:30] = np.array([0.0, 1.0, 0.0], np.float16)[:, None, None]
    job = infer.NetworkJob("crowdai", "b4", "seed0", "test", infer.InstanceParams(), PolygonParams("raw"), 100)
    timing = efficiency.postproc_time(job, [(1, probabilities), (2, probabilities), (3, probabilities)])
    assert timing["images"] == 3 and timing["instances_ms"] > 0 and timing["polygons_ms"] > 0
    assert timing["total_ms"] == pytest.approx(timing["instances_ms"] + timing["polygons_ms"])
    with pytest.raises(ValueError, match="no image"):
        efficiency.postproc_time(job, [])


def test_the_cpu_has_a_name():
    assert efficiency.cpu_name()
