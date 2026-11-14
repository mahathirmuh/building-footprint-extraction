"""Model factory and zero-initialised first convolution (docs/METHOD.md §2.5, §4). encoder_weights is always None."""

import copy

import pytest
import torch
from torch import nn

from densepoly.config import load_config
from densepoly.models.factory import CONTACT_OFF, build_model, input_channels, split_output
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
        ("b3m", "UnetPlusPlus", 3, 1),
        ("b3my", "UnetPlusPlus", 3, 1),
        ("full_y", "UnetPlusPlus", 3, 5),
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


# --- binary model with a θ head (B3θ, exploratory): 3 learned channels in the layout of FULL


def binary_theta_model():
    return build_model(load_config("crowdai", "b3t", NO_WEIGHTS)).eval()


def test_binary_theta_head_writes_the_layout_of_full():
    torch.manual_seed(0)
    model = binary_theta_model()
    with torch.inference_mode():
        output = model(torch.randn(2, 3, SIZE, SIZE))
    assert output.shape == (2, 5, SIZE, SIZE)
    assert (output[:, 0] == 0).all() and (output[:, 2] == CONTACT_OFF).all()
    logits, orientation = split_output(output)
    probabilities, building = torch.softmax(logits, dim=1), torch.sigmoid(output[:, 1])
    assert torch.allclose(probabilities[:, 1], building, atol=1e-6)
    assert torch.allclose(probabilities[:, 0], 1 - building, atol=1e-6)
    assert (probabilities[:, 2] == 0).all()  # the contact class is never predicted
    assert orientation.shape == (2, 2, SIZE, SIZE) and orientation.abs().max() <= 1


def test_the_contact_logit_stays_off_in_half_precision():
    logits = torch.tensor([[0.0, 30.0, CONTACT_OFF], [0.0, -30.0, CONTACT_OFF]]).half()
    assert torch.isfinite(logits).all()
    probabilities = torch.softmax(logits.float(), dim=1)
    assert (probabilities[:, 2] == 0).all() and torch.isfinite(probabilities).all()


def test_binary_theta_model_is_the_b3_network_with_two_more_output_channels():
    b3 = build_model(load_config("crowdai", "b3", NO_WEIGHTS))
    model = binary_theta_model()
    head = [m for m in model.segmentation_head.modules() if isinstance(m, nn.Conv2d)]
    assert [conv.out_channels for conv in head] == [3]
    count = lambda net: sum(p.numel() for p in net.parameters())  # noqa: E731
    assert count(model) - count(b3) == 2 * (head[0].in_channels * 9 + 1)
    twin = binary_theta_model()
    twin.load_state_dict(model.state_dict())
    x = torch.randn(1, 3, SIZE, SIZE)
    with torch.inference_mode():
        assert torch.equal(twin(x), model(x))


def test_binary_theta_head_gets_gradients_only_on_its_learned_channels():
    torch.manual_seed(0)
    model = binary_theta_model()
    output = model(torch.randn(2, 3, SIZE, SIZE))
    output.square().mean().backward()
    conv = next(m for m in model.segmentation_head.modules() if isinstance(m, nn.Conv2d))
    assert conv.weight.grad.abs().sum(dim=(1, 2, 3)).min() > 0


@pytest.mark.parametrize("overrides", [["model.head=two_heads"], ["model.classes=3"]])
def test_binary_theta_head_rejects_a_wrong_config(overrides):
    with pytest.raises(ValueError):
        build_model(load_config("crowdai", "b3t", [*NO_WEIGHTS, *overrides]))
