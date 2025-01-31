r"""Surrogate building blocks."""

__all__ = [
    "MaskedSurrogate",
    "get_surrogate",
]

import torch.nn as nn

from torch import Tensor
from typing import Dict, Optional, Sequence, Union

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
    arch: str,
    shape: Sequence[int],
    # Common
    emb_features: int,
    hid_channels: Union[int, Sequence[int]],
    hid_blocks: Union[int, Sequence[int]],
    attention_heads: Union[int, Dict[int, int]],
    dropout: Optional[float] = None,
    checkpointing: bool = False,
    identity_init: bool = True,
    # ViT
    qk_norm: bool = True,
    rope: bool = True,
    patch_size: Union[int, Sequence[int]] = 4,
    window_size: Optional[Sequence[int]] = None,
    # UNet
    kernel_size: Union[int, Sequence[int]] = 3,
    stride: Union[int, Sequence[int]] = 2,
    norm: str = "layer",
    periodic: bool = False,
    # Label
    label_features: int = 0,
    # Ignore
    name: str = None,
) -> nn.Module:
    r"""Instantiates a surrogate."""

    channels, *_ = shape

    if arch == "dit" or arch == "vit":
        backbone = ViT(
            in_channels=channels,
            out_channels=channels,
            cond_channels=channels,
            mod_features=emb_features,
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            attention_heads=attention_heads,
            qk_norm=qk_norm,
            rope=rope,
            spatial=len(shape) - 1,
            patch_size=patch_size,
            window_size=window_size,
            dropout=dropout,
            checkpointing=checkpointing,
        )
    elif arch == "unet":
        backbone = UNet(
            in_channels=channels,
            out_channels=channels,
            cond_channels=channels,
            mod_features=emb_features,
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            kernel_size=kernel_size,
            stride=stride,
            norm=norm,
            attention_heads=attention_heads,
            spatial=len(shape) - 1,
            periodic=periodic,
            dropout=dropout,
            checkpointing=checkpointing,
            identity_init=identity_init,
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
