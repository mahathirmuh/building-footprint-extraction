"""Experiment config → model (docs/METHOD.md §4).

Every model uses the same encoder, so that the steps from B1 to FULL change one thing at a time. FULL has one 5-channel
convolution as its head, which amounts to two heads (3 + 2 channels) sharing the decoder and does not depend on the
decoder API of smp, which changed between versions.

A binary model with a θ head (model.head = binary_theta; B3θ, exploratory) learns 3 channels (z, ŝ, ĉ) and writes them
in the layout of FULL, with class logits (0, z, CONTACT_OFF): the softmax of those is (1 − σ(z), σ(z), 0), so everything
behind the network reads its output like that of FULL, with a contact class that is never predicted.

A frozen network with a θ branch (model.head = frozen_theta; B3+θ, exploratory) is the trained binary network of
model.frozen_exp with every weight fixed, plus a small branch that reads its decoder features and predicts (ŝ, ĉ). Only
the branch learns, so the masks are those of the frozen network; the output has the same FULL layout.
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
from torch import nn

from densepoly.models.first_conv import expand_first_conv

ARCHITECTURES = {"unet": smp.Unet, "unetplusplus": smp.UnetPlusPlus}
FULL_CHANNELS = 5  # 3 class logits + (sin 4θ, cos 4θ)
HEADS = (None, "binary_theta", "frozen_theta")
THETA_BRANCH_WIDTH = 32  # channels of the θ branch on a frozen network
THETA_BRANCH_DILATIONS = (1, 2, 4, 8)  # its 3 × 3 convolutions: a receptive field of 31 px on the decoder features
CONTACT_OFF = -1.0e4  # contact logit of a binary model in the FULL layout: its softmax is exactly 0, also in float16


class BinaryThetaLayout(nn.Module):
    """(B, 3, H, W) with channels (z, ŝ, ĉ) → (B, 5, H, W) with channels (0, z, CONTACT_OFF, ŝ, ĉ)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 3:
            raise ValueError(f"a binary model with a θ head has 3 learned channels, got {x.shape[1]}")
        zero = torch.zeros_like(x[:, :1])
        return torch.cat([zero, x[:, :1], zero + CONTACT_OFF, x[:, 1:]], dim=1)


def theta_branch(in_channels: int) -> nn.Sequential:
    """Decoder features → the two orientation channels before the tanh: dilated 3 × 3 convolutions, then 1 × 1."""
    layers, channels = [], in_channels
    for dilation in THETA_BRANCH_DILATIONS:
        layers += [
            nn.Conv2d(channels, THETA_BRANCH_WIDTH, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(THETA_BRANCH_WIDTH),
            nn.ReLU(inplace=True),
        ]
        channels = THETA_BRANCH_WIDTH
    return nn.Sequential(*layers, nn.Conv2d(channels, 2, 1))


class FrozenThetaModel(nn.Module):
    """A binary segmentation network whose weights never change, and a trainable θ branch on its decoder features.

    base is the network without its head (its forward returns the decoder features), head its own 1-channel head.
    Both stay in eval mode whatever mode the model is in, so that their BatchNorm statistics are those of the trained
    network also while the branch trains: the building logit z is exactly that of the frozen network.
    """

    def __init__(self, network: nn.Module):
        super().__init__()
        self.head = network.segmentation_head
        network.segmentation_head = nn.Identity()
        self.base = network
        convs = [module for module in self.head.modules() if isinstance(module, nn.Conv2d)]
        if len(convs) != 1 or convs[0].out_channels != 1:
            raise ValueError("a frozen network with a θ branch needs a binary network (one 1-channel head)")
        self.theta = theta_branch(convs[0].in_channels)
        for frozen in (self.base, self.head):
            for parameter in frozen.parameters():
                parameter.requires_grad_(False)
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        self.head.eval()
        return self

    def load_frozen(self, state: dict) -> None:
        """Weights of the trained binary network (the "model" of its checkpoint) into the frozen part."""
        prefix = "segmentation_head."
        self.base.load_state_dict({key: value for key, value in state.items() if not key.startswith(prefix)})
        self.head.load_state_dict({key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.base(x)
            building = self.head(features)
        zero = torch.zeros_like(building)
        return torch.cat([zero, building, zero + CONTACT_OFF, self.theta(features)], dim=1)


def model_head(cfg) -> str | None:
    head = cfg.model.get("head")
    if head not in HEADS:
        raise ValueError(f"unknown model head {head!r}; expected one of {HEADS}")
    return head


def input_channels(cfg) -> int:
    return 3 + len(cfg.input.priors)


def build_model(cfg) -> nn.Module:
    """The model of an experiment; models with priors are built with 3 channels, then expanded with zero weights."""
    arch = str(cfg.model.arch)
    if arch not in ARCHITECTURES:
        raise ValueError(f"no network for architecture {arch!r}; expected one of {sorted(ARCHITECTURES)}")
    head, classes = model_head(cfg), int(cfg.model.classes)
    if head is not None and classes != FULL_CHANNELS:
        raise ValueError(f"head {head} writes the {FULL_CHANNELS} channels of FULL; model.classes is {classes}")
    learned = {None: classes, "binary_theta": 3, "frozen_theta": 1}[head]
    model = ARCHITECTURES[arch](
        str(cfg.model.encoder), encoder_weights=cfg.model.encoder_weights, in_channels=3, classes=learned
    )
    if head == "binary_theta":
        model.segmentation_head = nn.Sequential(model.segmentation_head, BinaryThetaLayout())
    n_in = input_channels(cfg)
    if n_in > 3:
        expand_first_conv(model, n_in)
    if head == "frozen_theta":
        if not cfg.model.get("frozen_exp"):
            raise ValueError("head frozen_theta needs model.frozen_exp, the experiment of the frozen network")
        return FrozenThetaModel(model)
    return model


def split_output(output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """(logits, orientation) of a model output: FULL gives 3 class logits and tanh(ŝ, ĉ), the others logits only."""
    if output.shape[1] == FULL_CHANNELS:
        return output[:, :3], torch.tanh(output[:, 3:])
    return output, None
