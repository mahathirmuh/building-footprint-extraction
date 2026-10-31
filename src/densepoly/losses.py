"""Losses of every model (docs/METHOD.md §5).

- Binary (B1-B3): Σ w·BCE / non-ignored pixels + binary Dice, with w = 1 on B1-B2 and the boundary weight on B3.
- Three classes (B4): Σ w·CE / non-ignored pixels + multiclass Dice over classes 1 and 2.
- FULL: the loss of B4 + λ·L_θ, L_θ = Σ_{p∈V} [(ŝ − S)² + (ĉ − C)²] / max(|V|, 1), so buildings too small for a θ and
  background pixels add nothing to it.

Padding pixels (label 255) add nothing to any model. The whole loss runs in float32, whatever the autocast context of
the model: softmax and Dice are not safe in float16.
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch import nn

from densepoly.models.factory import split_output

KINDS = ("bce_dice", "ce_dice", "ce_dice_theta")


class SegmentationLoss(nn.Module):
    """forward(model output, batch) gives the total "loss" and its terms "ce" (BCE on binary models), "dice", "theta".
    A batch holds classes (B, H, W) long, weight (B, H, W), and for FULL sin4, cos4 and valid (B, H, W)."""

    def __init__(self, cfg):
        super().__init__()
        self.kind = str(cfg.loss.type)
        if self.kind not in KINDS:
            raise ValueError(f"unknown loss {self.kind!r}; expected one of {KINDS}")
        self.ignore = int(cfg.data.ignore_index)
        self.edge_weight = bool(cfg.loss.edge_weight)
        self.theta_lambda = float(cfg.loss.theta_lambda)
        self.register_buffer("class_weight", torch.tensor([1.0, 1.0, float(cfg.loss.contact_class_weight)]))
        if self.kind == "bce_dice":
            self.dice = smp.losses.DiceLoss(mode="binary", ignore_index=self.ignore)
        else:
            self.dice = smp.losses.DiceLoss(mode="multiclass", classes=[1, 2], ignore_index=self.ignore)

    def forward(self, output: torch.Tensor, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        with torch.autocast(output.device.type, enabled=False):
            logits, orientation = split_output(output.float())
            classes = batch["classes"]
            keep = classes != self.ignore
            if self.kind == "bce_dice":
                target = torch.where(keep, (classes > 0).long(), classes)  # the BCE target is y == 1, never 255
                pixel_loss = F.binary_cross_entropy_with_logits(logits[:, 0], (target == 1).float(), reduction="none")
                dice = self.dice(logits, target)
            else:
                pixel_loss = F.cross_entropy(
                    logits, classes, weight=self.class_weight, ignore_index=self.ignore, reduction="none"
                )
                dice = self.dice(logits, classes)
            weight = batch["weight"].float() if self.edge_weight else torch.ones_like(pixel_loss)
            ce = (weight * keep * pixel_loss).sum() / keep.sum().clamp(min=1)
            theta = self.theta_term(orientation, batch) if self.kind == "ce_dice_theta" else torch.zeros_like(ce)
            return {"loss": ce + dice + self.theta_lambda * theta, "ce": ce, "dice": dice, "theta": theta}

    @staticmethod
    def theta_term(orientation: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        valid = batch["valid"]
        squared = (orientation[:, 0] - batch["sin4"].float()) ** 2 + (orientation[:, 1] - batch["cos4"].float()) ** 2
        return (squared * valid).sum() / valid.sum().clamp(min=1)
