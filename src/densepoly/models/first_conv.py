"""Zero-initialised expansion of the first convolution (docs/METHOD.md §2.5, contribution N3).

smp's in_channels=5 fills the new channels with cyclic copies of the RGB weights and rescales every weight by 3 / 5, so
the pretrained RGB weights change. Here the model is built with 3 channels and the first convolution gets extra channels
with zero weights: the expanded model starts out as the RGB model, and the gradients on those weights are not zero
because their inputs are not, so it learns by itself how much to use the priors.
"""

from __future__ import annotations

import torch
from torch import nn


def expand_first_conv(model: nn.Module, n_in: int) -> None:
    """Extend the first 3-channel convolution of model.encoder to n_in channels, in place. The convolution is kept
    rather than rebuilt: the EfficientNet stem of smp is an nn.Conv2d subclass that pads in its forward."""
    if n_in <= 3:
        raise ValueError(f"expected more than 3 input channels, got {n_in}")
    conv = next(m for m in model.encoder.modules() if isinstance(m, nn.Conv2d) and m.in_channels == 3)
    old = conv.weight.detach()
    new = torch.zeros(old.shape[0], n_in, *old.shape[2:], dtype=old.dtype, device=old.device)
    new[:, :3] = old
    conv.in_channels = n_in
    conv.weight = nn.Parameter(new)
    model.encoder._in_channels = n_in
    model.encoder._out_channels = [n_in, *model.encoder._out_channels[1:]]
