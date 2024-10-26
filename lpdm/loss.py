r"""Losses and criterions."""

__all__ = [
    "WeightedLoss",
]

import torch
import torch.nn as nn

from torch import Tensor
from typing import Sequence


class WeightedLoss(nn.Module):
    r"""Creates a weighted loss module."""

    def __init__(
        self,
        losses: Sequence[str] = ["mse"],  # noqa: B006
        weights: Sequence[float] = [1.0],  # noqa: B006
    ):
        super().__init__()

        assert len(losses) == len(weights)

        LOSSES = {
            "mae": mae,
            "mse": mse,
        }

        self.losses = [LOSSES[key] for key in losses]
        self.register_buffer("weights", torch.as_tensor(weights))

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        values = torch.stack([loss(x, y) for loss in self.losses])

        return torch.vdot(self.weights, values)


def mae(x: Tensor, y: Tensor) -> Tensor:
    return (x - y).abs().mean()


def mse(x: Tensor, y: Tensor) -> Tensor:
    return (x - y).square().mean()
