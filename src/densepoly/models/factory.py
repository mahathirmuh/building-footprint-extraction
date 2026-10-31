"""Experiment config → model (docs/METHOD.md §4).

Every model uses the same encoder, so that the steps from B1 to FULL change one thing at a time. FULL has one 5-channel
convolution as its head, which amounts to two heads (3 + 2 channels) sharing the decoder and does not depend on the
decoder API of smp, which changed between versions.
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
from torch import nn

from densepoly.models.first_conv import expand_first_conv

ARCHITECTURES = {"unet": smp.Unet, "unetplusplus": smp.UnetPlusPlus}
FULL_CHANNELS = 5  # 3 class logits + (sin 4θ, cos 4θ)


def input_channels(cfg) -> int:
    return 3 + len(cfg.input.priors)


def build_model(cfg) -> nn.Module:
    """The model of an experiment; models with priors are built with 3 channels, then expanded with zero weights."""
    arch = str(cfg.model.arch)
    if arch not in ARCHITECTURES:
        raise ValueError(f"no network for architecture {arch!r}; expected one of {sorted(ARCHITECTURES)}")
    model = ARCHITECTURES[arch](
        str(cfg.model.encoder), encoder_weights=cfg.model.encoder_weights, in_channels=3, classes=int(cfg.model.classes)
    )
    n_in = input_channels(cfg)
    if n_in > 3:
        expand_first_conv(model, n_in)
    return model


def split_output(output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """(logits, orientation) of a model output: FULL gives 3 class logits and tanh(ŝ, ĉ), the others logits only."""
    if output.shape[1] == FULL_CHANNELS:
        return output[:, :3], torch.tanh(output[:, 3:])
    return output, None
