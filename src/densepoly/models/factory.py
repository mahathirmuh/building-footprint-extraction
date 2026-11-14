"""Experiment config → model (docs/METHOD.md §4).

Every model uses the same encoder, so that the steps from B1 to FULL change one thing at a time. FULL has one 5-channel
convolution as its head, which amounts to two heads (3 + 2 channels) sharing the decoder and does not depend on the
decoder API of smp, which changed between versions.

A binary model with a θ head (model.head = binary_theta; B3θ, exploratory) learns 3 channels (z, ŝ, ĉ) and writes them
in the layout of FULL, with class logits (0, z, CONTACT_OFF): the softmax of those is (1 − σ(z), σ(z), 0), so everything
behind the network reads its output like that of FULL, with a contact class that is never predicted.
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
from torch import nn

from densepoly.models.first_conv import expand_first_conv

ARCHITECTURES = {"unet": smp.Unet, "unetplusplus": smp.UnetPlusPlus}
FULL_CHANNELS = 5  # 3 class logits + (sin 4θ, cos 4θ)
HEADS = (None, "binary_theta")
CONTACT_OFF = -1.0e4  # contact logit of a binary model in the FULL layout: its softmax is exactly 0, also in float16


class BinaryThetaLayout(nn.Module):
    """(B, 3, H, W) with channels (z, ŝ, ĉ) → (B, 5, H, W) with channels (0, z, CONTACT_OFF, ŝ, ĉ)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 3:
            raise ValueError(f"a binary model with a θ head has 3 learned channels, got {x.shape[1]}")
        zero = torch.zeros_like(x[:, :1])
        return torch.cat([zero, x[:, :1], zero + CONTACT_OFF, x[:, 1:]], dim=1)


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
    if head == "binary_theta" and classes != FULL_CHANNELS:
        raise ValueError(f"head binary_theta writes the {FULL_CHANNELS} channels of FULL; model.classes is {classes}")
    learned = 3 if head == "binary_theta" else classes
    model = ARCHITECTURES[arch](
        str(cfg.model.encoder), encoder_weights=cfg.model.encoder_weights, in_channels=3, classes=learned
    )
    if head == "binary_theta":
        model.segmentation_head = nn.Sequential(model.segmentation_head, BinaryThetaLayout())
    n_in = input_channels(cfg)
    if n_in > 3:
        expand_first_conv(model, n_in)
    return model


def split_output(output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """(logits, orientation) of a model output: FULL gives 3 class logits and tanh(ŝ, ĉ), the others logits only."""
    if output.shape[1] == FULL_CHANNELS:
        return output[:, :3], torch.tanh(output[:, 3:])
    return output, None
