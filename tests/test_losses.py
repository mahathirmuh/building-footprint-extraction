"""Losses of every model (docs/METHOD.md §5): known values, padding, weights, and training on a synthetic scene."""

import math

import pytest
import torch
from torch import nn

from densepoly.config import load_config
from densepoly.labels.targets import targets_from_config
from densepoly.losses import SegmentationLoss
from densepoly.models.factory import build_model
from tests.fixtures.synthetic import make_scene

IGNORE = 255
EXPERIMENTS = ("b2", "b3", "b4", "full_p0")  # bce_dice, weighted bce_dice, ce_dice, ce_dice_theta


def criterion(exp, *overrides):
    return SegmentationLoss(load_config("crowdai", exp, overrides))


def output_channels(exp):
    return {"b2": 1, "b3": 1, "b4": 3, "full_p0": 5}[exp]


def toy_batch():
    """Eight pixels in one row-column grid: 3 background, 2 building, 1 contact, 2 padding."""
    classes = torch.tensor([[[0, 0, 0, 1], [1, 2, IGNORE, IGNORE]]])
    return {
        "classes": classes,
        "weight": torch.ones(classes.shape) * (classes != IGNORE),
        "sin4": torch.tensor([[[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]]]),
        "cos4": torch.tensor([[[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]),
        "valid": torch.tensor([[[False, False, False, True], [True, True, False, False]]]),
    }


def test_uniform_three_class_output_has_the_known_ce_and_dice():
    batch = toy_batch()
    terms = criterion("b4")(torch.zeros(1, 3, 2, 4), batch)
    assert terms["ce"].item() == pytest.approx(math.log(3), abs=1e-6)  # padding is left out of the mean
    # On 6 pixels with p = 1/3 everywhere: Dice = 2 Σpy / (Σp + Σy), n_1 = 2 and n_2 = 1 of class 1 and 2.
    dice_1, dice_2 = 2 * (2 / 3) / (2 + 2), 2 * (1 / 3) / (2 + 1)
    assert terms["dice"].item() == pytest.approx(1 - (dice_1 + dice_2) / 2, abs=1e-6)
    assert terms["theta"].item() == 0
    assert terms["loss"].item() == pytest.approx(terms["ce"].item() + terms["dice"].item(), abs=1e-6)


def test_uniform_binary_output_has_the_known_bce_and_dice():
    terms = criterion("b2")(torch.zeros(1, 1, 2, 4), toy_batch())
    assert terms["ce"].item() == pytest.approx(math.log(2), abs=1e-6)
    assert terms["dice"].item() == pytest.approx(1 - 2 * (3 / 2) / (3 + 3), abs=1e-6)  # contact counts as building


def test_theta_term_is_the_mean_squared_error_over_valid_pixels():
    output = torch.zeros(1, 5, 2, 4)  # tanh(0) = 0: squared errors 1, 1, 1 on the three valid pixels
    terms = criterion("full_p0", "loss.theta_lambda=2.0")(output, toy_batch())
    assert terms["theta"].item() == pytest.approx(1.0)
    assert terms["loss"].item() == pytest.approx(terms["ce"].item() + terms["dice"].item() + 2.0, abs=1e-6)


def test_theta_term_is_zero_without_valid_pixels():
    batch = toy_batch()
    batch["valid"] = torch.zeros_like(batch["valid"])
    output = torch.ones(1, 5, 2, 4, requires_grad=True)
    terms = criterion("full_p0")(output, batch)
    assert terms["theta"].item() == 0
    terms["loss"].backward()
    assert torch.isfinite(output.grad).all()


@pytest.mark.parametrize("exp", EXPERIMENTS)
def test_padding_adds_nothing_to_any_term(exp):
    batch = toy_batch()
    n = output_channels(exp)
    torch.manual_seed(0)
    output = torch.randn(1, n, 2, 4)
    changed = output.clone()
    changed[:, :, batch["classes"][0] == IGNORE] = 50.0
    loss = criterion(exp)
    first, second = loss(output, batch), loss(changed, batch)
    assert all(first[key].item() == pytest.approx(second[key].item(), abs=1e-6) for key in first)
    # ... and the padded pixels get no gradient
    output.requires_grad_()
    loss(output, batch)["loss"].backward()
    assert output.grad[:, :, batch["classes"][0] == IGNORE].abs().sum() == 0


def test_theta_ignores_the_orientation_outside_valid_pixels():
    batch = toy_batch()
    torch.manual_seed(0)
    output = torch.randn(1, 5, 2, 4)
    changed = output.clone()
    changed[:, 3:, ~batch["valid"][0]] = 7.0
    loss = criterion("full_p0")
    assert loss(output, batch)["theta"].item() == pytest.approx(loss(changed, batch)["theta"].item(), abs=1e-6)


@pytest.mark.parametrize(("exp", "n"), [("b3", 1), ("b4", 3)])
def test_the_boundary_weight_scales_the_pixel_term_only_when_enabled(exp, n):
    batch = toy_batch()
    heavy = {**batch, "weight": batch["weight"] * 3}
    torch.manual_seed(0)
    output = torch.randn(1, n, 2, 4)
    weighted = criterion(exp)
    assert weighted(output, heavy)["ce"].item() == pytest.approx(3 * weighted(output, batch)["ce"].item(), rel=1e-6)
    plain = criterion(exp, "loss.edge_weight=false")
    assert plain(output, heavy)["ce"].item() == pytest.approx(plain(output, batch)["ce"].item(), rel=1e-6)


def test_the_contact_class_weight_scales_contact_pixels_only():
    batch = toy_batch()
    output = torch.zeros(1, 3, 2, 4)
    base = criterion("b4")(output, batch)["ce"].item()
    weighted = criterion("b4", "loss.contact_class_weight=5.0")(output, batch)["ce"].item()
    assert weighted == pytest.approx(base + 4 * math.log(3) / 6, abs=1e-6)  # one contact pixel among 6
    no_contact = {**batch, "classes": batch["classes"].where(batch["classes"] != 2, torch.tensor(1))}
    assert criterion("b4", "loss.contact_class_weight=5.0")(output, no_contact)["ce"].item() == pytest.approx(base)


def test_the_loss_runs_in_float32_for_half_outputs_and_inside_autocast():
    loss = criterion("full_p0")
    output = torch.zeros(1, 5, 2, 4).half()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert loss(output, toy_batch())["loss"].dtype == torch.float32


def test_inference_only_experiments_have_no_loss():
    with pytest.raises(ValueError):
        criterion("b5")


def tiny_batch(cfg):
    """A 96 × 96 crop of a synthetic scene with its labels, and the RGB input."""
    scene = make_scene(3)
    image = torch.from_numpy(scene.image[:96, :96].copy()).permute(2, 0, 1).float()[None] / 255 - 0.5
    targets = targets_from_config(scene.instance_map[:96, :96], cfg.labels)
    batch = {
        "classes": torch.from_numpy(targets.classes.copy()).long()[None],
        "weight": torch.from_numpy(targets.weight.copy())[None],
        "sin4": torch.from_numpy(targets.sin4.copy())[None],
        "cos4": torch.from_numpy(targets.cos4.copy())[None],
        "valid": torch.from_numpy(targets.valid.copy())[None],
    }
    return image, batch


@pytest.mark.parametrize("exp", EXPERIMENTS)
def test_the_loss_decreases_on_a_synthetic_scene(exp):
    cfg = load_config("crowdai", exp)
    loss = SegmentationLoss(cfg)
    image, batch = tiny_batch(cfg)
    torch.manual_seed(0)
    n = output_channels(exp)
    net = nn.Sequential(nn.Conv2d(3, 16, 5, padding=2), nn.ReLU(), nn.Conv2d(16, 16, 5, padding=2), nn.ReLU(),
                        nn.Conv2d(16, n, 1))  # fmt: skip
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    values = []
    for _ in range(80):
        optimizer.zero_grad()
        terms = loss(net(image), batch)
        terms["loss"].backward()
        optimizer.step()
        values.append(terms["loss"].item())
    assert values[-1] < 0.8 * values[0]


def test_the_weights_of_the_prior_channels_learn_from_the_loss():
    cfg = load_config("crowdai", "full_p2", ["model.encoder_weights=null"])
    torch.manual_seed(0)
    model = build_model(cfg).train()
    image, batch = tiny_batch(cfg)
    x = torch.cat([image, torch.rand(1, 2, 96, 96)], dim=1).repeat(2, 1, 1, 1)
    batch = {key: value.repeat(2, 1, 1) for key, value in batch.items()}
    SegmentationLoss(cfg)(model(x), batch)["loss"].backward()
    first = next(m for m in model.encoder.modules() if isinstance(m, nn.Conv2d))
    assert first.weight.grad[:, 3:].abs().sum() > 0
