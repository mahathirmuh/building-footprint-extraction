"""Model factory and zero-initialised first convolution (docs/METHOD.md §2.5, §4). encoder_weights is always None."""

import copy

import pytest
import torch
from torch import nn

from densepoly.config import load_config
from densepoly.models.factory import build_model, input_channels, split_output
from densepoly.models.first_conv import expand_first_conv

NO_WEIGHTS = ["model.encoder_weights=null"]
SIZE = 64


def rgb_model(classes=5):
    return build_model(load_config("crowdai", "full_p0", [*NO_WEIGHTS, f"model.classes={classes}"])).eval()


def first_conv(model):
    return next(m for m in model.encoder.modules() if isinstance(m, nn.Conv2d))


@pytest.mark.parametrize("n_in", [4, 5])
def test_expanded_model_equals_the_rgb_model_for_any_prior_input(n_in):
    torch.manual_seed(0)
    rgb = rgb_model()
    expanded = copy.deepcopy(rgb)
    expand_first_conv(expanded, n_in)
    expanded.eval()
    x = torch.randn(2, n_in, SIZE, SIZE)
    with torch.inference_mode():
        assert torch.allclose(expanded(x), rgb(x[:, :3]), atol=1e-5)
        changed = x.clone()
        changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 100
        assert torch.allclose(expanded(changed), rgb(x[:, :3]), atol=1e-5)


def test_expansion_keeps_the_rgb_weights_and_adds_zero_channels():
    torch.manual_seed(0)
    model = rgb_model()
    before = first_conv(model).weight.detach().clone()
    expand_first_conv(model, 5)
    conv = first_conv(model)
    assert conv.in_channels == 5 and conv.weight.shape[1] == 5
    assert torch.equal(conv.weight[:, :3], before)
    assert conv.weight[:, 3:].abs().sum() == 0
    assert model.encoder.out_channels[0] == 5 and model.encoder._in_channels == 5


def test_the_prior_weights_get_a_gradient_although_they_start_at_zero():
    torch.manual_seed(0)
    model = rgb_model()
    expand_first_conv(model, 5)
    model.eval()
    model(torch.randn(2, 5, SIZE, SIZE)).square().mean().backward()
    assert first_conv(model).weight.grad[:, 3:].abs().sum() > 0


def test_expansion_refuses_a_smaller_or_equal_channel_count():
    with pytest.raises(ValueError):
        expand_first_conv(rgb_model(), 3)


@pytest.mark.parametrize(
    ("exp", "arch", "n_in", "n_out"),
    [
        ("b1", "Unet", 3, 1),
        ("b2", "UnetPlusPlus", 3, 1),
        ("b3", "UnetPlusPlus", 3, 1),
        ("b4", "UnetPlusPlus", 3, 3),
        ("b5", "UnetPlusPlus", 3, 3),
        ("b5p", "UnetPlusPlus", 3, 5),
        ("full_p0", "UnetPlusPlus", 3, 5),
        ("full_p1", "UnetPlusPlus", 4, 5),
        ("full_p2", "UnetPlusPlus", 5, 5),
    ],
)
def test_every_experiment_gets_its_architecture_and_channels(exp, arch, n_in, n_out):
    cfg = load_config("crowdai", exp, NO_WEIGHTS)
    model = build_model(cfg).eval()
    assert type(model).__name__ == arch and input_channels(cfg) == n_in
    with torch.inference_mode():
        assert model(torch.zeros(1, n_in, SIZE, SIZE)).shape == (1, n_out, SIZE, SIZE)


def test_the_classical_baseline_has_no_network():
    with pytest.raises(ValueError):
        build_model(load_config("crowdai", "b0", NO_WEIGHTS))


def test_input_sides_must_be_multiples_of_32():
    with pytest.raises(RuntimeError):
        rgb_model(3)(torch.zeros(1, 3, 60, 60))


def test_split_output_separates_logits_from_the_bounded_orientation():
    output = torch.randn(2, 5, 8, 8) * 10
    logits, orientation = split_output(output)
    assert torch.equal(logits, output[:, :3])
    assert torch.equal(orientation, torch.tanh(output[:, 3:]))
    assert orientation.abs().max() <= 1
    three = torch.randn(2, 3, 8, 8)
    assert split_output(three)[0] is three and split_output(three)[1] is None
