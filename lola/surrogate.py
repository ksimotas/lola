r"""Surrogate building blocks."""

__all__ = [
    "MaskedSurrogate",
    "RegressionLoss",
    "get_surrogate",
]

import torch
import torch.nn as nn

from einops import rearrange
from torch import Tensor
from typing import Optional, Sequence

from .nn.unet import UNet
from .nn.vit import ViT


class MaskedSurrogate(nn.Module):
    r"""Creates a masked surrogate module.

    Arguments:
        backbone: A surrogate backbone.
        label_embedding: Optional[nn.Module] = None,
    """

    def __init__(
        self,
        backbone: nn.Module,
        label_embedding: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.backbone = backbone
        self.label_embedding = label_embedding

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        label: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        kwargs.setdefault("cond", mask.expand_as(x))

        if label is None:
            emb = None
        else:
            emb = self.label_embedding(label)

        return self.backbone(x * mask, emb, **kwargs)


class RegressionLoss(nn.Module):
    r"""Creates a weighted regression loss module."""

    def __init__(
        self,
        losses: Sequence[str] = ["mse"],  # noqa: B006
        weights: Sequence[float] = [1.0],  # noqa: B006
    ):
        super().__init__()

        assert len(losses) == len(weights)

        self.losses = list(losses)
        self.register_buffer("weights", torch.as_tensor(weights))

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The target tensor :math:`x`, with shape :math:`(B, C, ...)`.
            y: The output tensor :math:`y`, with shape :math:`(B, C, ...)`.

        Returns:
            The weighted loss.
        """

        values = []

        for loss in self.losses:
            if loss == "mse":
                l = (x - y).square().mean()
            elif loss == "mae":
                l = (x - y).abs().mean()
            elif loss == "vmse":
                x = rearrange(x, "B C ... -> B C (...)")
                y = rearrange(y, "B C ... -> B C (...)")
                l = (x - y).square().mean(dim=2) / (x.var(dim=2) + 1e-2)
                l = l.mean()
            elif loss == "vrmse":
                x = rearrange(x, "B C ... -> B C (...)")
                y = rearrange(y, "B C ... -> B C (...)")
                l = (x - y).square().mean(dim=2) / (x.var(dim=2) + 1e-2)
                l = torch.sqrt(l).mean()
            else:
                raise ValueError(f"unknown loss '{loss}'.")

            values.append(l)

        values = torch.stack(values)

        return torch.vdot(self.weights, values)


def get_surrogate(
    channels: int,
    # Arch
    arch: Optional[str] = None,
    emb_features: int = 256,
    label_features: int = 0,
    # Ignore
    name: str = None,
    # Passthrough
    **kwargs,
) -> nn.Module:
    r"""Instantiates a surrogate."""

    if arch in (None, "dit", "vit"):
        backbone = ViT(
            in_channels=channels,
            out_channels=channels,
            cond_channels=channels,
            mod_features=emb_features if label_features > 0 else 0,
            **kwargs,
        )
    elif arch == "unet":
        backbone = UNet(
            in_channels=channels,
            out_channels=channels,
            cond_channels=channels,
            mod_features=emb_features if label_features > 0 else 0,
            **kwargs,
        )
    else:
        raise NotImplementedError()

    if label_features > 0:
        label_embedding = nn.Sequential(
            nn.Linear(label_features, emb_features),
            nn.SiLU(),
            nn.Linear(emb_features, emb_features),
        )
    else:
        label_embedding = None

    surrogate = MaskedSurrogate(
        backbone=backbone,
        label_embedding=label_embedding,
    )

    return surrogate
