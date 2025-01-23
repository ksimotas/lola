r"""Surrogate building blocks."""

__all__ = [
    "MaskedSurrogate",
    "get_surrogate",
]

import torch.nn as nn

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


def get_surrogate(
    shape: Sequence[int],
    label_features: int = 0,
    # Arch
    arch: Optional[str] = None,
    emb_features: int = 256,
    # Ignore
    name: str = None,
    # Passthrough
    **kwargs,
) -> nn.Module:
    r"""Instantiates a surrogate."""

    channels, *_ = shape

    if arch in (None, "dit", "vit"):
        backbone = ViT(
            in_channels=channels,
            out_channels=channels,
            cond_channels=channels,
            mod_features=emb_features if label_features > 0 else 0,
            spatial=len(shape) - 1,
            **kwargs,
        )
    elif arch == "unet":
        backbone = UNet(
            in_channels=channels,
            out_channels=channels,
            cond_channels=channels,
            mod_features=emb_features if label_features > 0 else 0,
            spatial=len(shape) - 1,
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
