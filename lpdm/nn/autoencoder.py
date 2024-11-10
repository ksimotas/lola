r"""Auto-encoder building blocks."""

__all__ = [
    "ResBlock",
    "Encoder",
    "Decoder",
    "AutoEncoder",
]

import math
import torch
import torch.nn as nn

from torch import Tensor
from torch.utils.checkpoint import checkpoint
from typing import Dict, Optional, Sequence, Tuple, Union

from .layers import (
    ConvNd,
    LayerNorm,
    Patchify,
    SelfAttentionNd,
    Unpatchify,
)


class Residual(nn.Sequential):
    def forward(self, x: Tensor) -> Tensor:
        return x + super().forward(x)


class ResBlock(nn.Module):
    r"""Creates a residual block module.

    Arguments:
        channels: The number of channels :math:`C`.
        norm: The kind of normalization.
        groups: The number of groups in :class:`torch.nn.GroupNorm` layers.
        attention_heads: The number of attention heads.
        spatial: The number of spatial dimensions :math:`N`.
        dropout: The dropout rate in :math:`[0, 1]`.
        kwargs: Keyword arguments passed to :class:`torch.nn.Conv2d`.
    """

    def __init__(
        self,
        channels: int,
        norm: str = "group",
        groups: int = 16,
        attention_heads: Optional[int] = None,
        spatial: int = 2,
        dropout: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()

        # Attention
        if attention_heads is None:
            self.attn = nn.Identity()
        else:
            self.attn = Residual(
                SelfAttentionNd(channels, heads=attention_heads),
            )

        # Block
        self.block = nn.Sequential(
            ConvNd(channels, channels, spatial=spatial, **kwargs),
            nn.SiLU(),
            nn.Identity() if dropout is None else nn.Dropout(dropout),
            ConvNd(channels, channels, spatial=spatial, **kwargs),
        )

        if norm == "group":
            self.block.insert(
                0,
                nn.GroupNorm(
                    num_groups=min(groups, channels),
                    num_channels=channels,
                    affine=False,
                ),
            )
        elif norm == "layer":
            self.block.insert(0, LayerNorm(dim=-spatial - 1))
        else:
            raise NotImplementedError()

        self.register_buffer("out_scale", torch.as_tensor(math.sqrt(1 / 2)))

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor, with shape :math:`(B, C, L_1, ..., L_N)`.

        Returns:
            The output tensor, with shape :math:`(B, C, L_1, ..., L_N)`.
        """

        y = self.attn(x)
        y = self.block(y)

        return self.out_scale * (x + y)


class Encoder(nn.Module):
    r"""Creates an encoder module.

    Arguments:
        in_channels: The number of input channels :math:`C_i`.
        out_channels: The number of output channels :math:`C_o`.
        hid_channels: The numbers of channels at each depth.
        hid_blocks: The numbers of hidden blocks at each depth.
        kernel_size: The kernel size of all convolutions.
        stride: The stride of the downsampling convolutions.
        pixel_shuffle: Whether to use pixel shuffling or not.
        norm: The kind of normalization.
        attention_heads: The number of attention heads at each depth.
        spatial: The number of spatial dimensions.
        periodic: Whether the spatial dimensions are periodic or not.
        dropout: The dropout rate in :math:`[0, 1]`.
        checkpointing: Whether to use gradient checkpointing or not.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 2,
        pixel_shuffle: bool = False,
        linear_out: bool = True,
        norm: str = "group",
        attention_heads: Dict[int, int] = {},  # noqa: B006
        spatial: int = 2,
        periodic: bool = False,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
    ):
        super().__init__()

        assert len(hid_blocks) == len(hid_channels)
        assert linear_out, "non-linear output projection is not supported"

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * spatial

        if isinstance(stride, int):
            stride = [stride] * spatial

        kwargs = dict(
            kernel_size=tuple(kernel_size),
            padding=tuple(k // 2 for k in kernel_size),
            padding_mode="circular" if periodic else "zeros",
        )

        self.descent = nn.ModuleList()

        for i, num_blocks in enumerate(hid_blocks):
            blocks = nn.ModuleList()

            if i > 0:
                if pixel_shuffle:
                    blocks.append(
                        nn.Sequential(
                            Patchify(patch_size=stride),
                            ConvNd(
                                hid_channels[i - 1] * math.prod(stride),
                                hid_channels[i],
                                spatial=spatial,
                                identity_init=True,
                                **kwargs,
                            ),
                        )
                    )
                else:
                    blocks.append(
                        ConvNd(
                            hid_channels[i - 1],
                            hid_channels[i],
                            spatial=spatial,
                            stride=stride,
                            identity_init=True,
                            **kwargs,
                        )
                    )
            else:
                blocks.append(ConvNd(in_channels, hid_channels[i], spatial=spatial, **kwargs))

            for _ in range(num_blocks):
                blocks.append(
                    ResBlock(
                        hid_channels[i],
                        norm=norm,
                        attention_heads=attention_heads.get(i, None),
                        dropout=dropout,
                        spatial=spatial,
                        **kwargs,
                    )
                )

            if i + 1 == len(hid_blocks):
                blocks.append(ConvNd(hid_channels[i], out_channels, spatial=spatial, **kwargs))

            self.descent.append(blocks)

        self.checkpointing = checkpointing

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor, with shape :math:`(B, C_i, L_1, ..., L_N)`.

        Returns:
            The output tensor, with shape :math:`(B, C_o, L_1 / 2^D, ..., L_N  / 2^D)`.
        """

        for blocks in self.descent:
            for block in blocks:
                if self.checkpointing and isinstance(block, ResBlock):
                    x = checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)

        return x


class Decoder(nn.Module):
    r"""Creates a decoder module.

    Arguments:
        in_channels: The number of input channels :math:`C_i`.
        out_channels: The number of output channels :math:`C_o`.
        hid_channels: The numbers of channels at each depth.
        hid_blocks: The numbers of hidden blocks at each depth.
        kernel_size: The kernel size of all convolutions.
        stride: The stride of the downsampling convolutions.
        pixel_shuffle: Whether to use pixel shuffling or not.
        norm: The kind of normalization.
        attention_heads: The number of attention heads at each depth.
        spatial: The number of spatial dimensions.
        periodic: Whether the spatial dimensions are periodic or not.
        dropout: The dropout rate in :math:`[0, 1]`.
        checkpointing: Whether to use gradient checkpointing or not.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 2,
        pixel_shuffle: bool = False,
        linear_out: bool = True,
        norm: str = "group",
        attention_heads: Dict[int, int] = {},  # noqa: B006
        spatial: int = 2,
        periodic: bool = False,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
    ):
        super().__init__()

        assert len(hid_blocks) == len(hid_channels)
        assert linear_out, "non-linear output projection is not supported"

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * spatial

        if isinstance(stride, int):
            stride = [stride] * spatial

        kwargs = dict(
            kernel_size=tuple(kernel_size),
            padding=tuple(k // 2 for k in kernel_size),
            padding_mode="circular" if periodic else "zeros",
        )

        self.ascent = nn.ModuleList()

        for i, num_blocks in reversed(list(enumerate(hid_blocks))):
            blocks = nn.ModuleList()

            if i + 1 == len(hid_blocks):
                blocks.append(ConvNd(in_channels, hid_channels[i], spatial=spatial, **kwargs))

            for _ in range(num_blocks):
                blocks.append(
                    ResBlock(
                        hid_channels[i],
                        norm=norm,
                        attention_heads=attention_heads.get(i, None),
                        spatial=spatial,
                        dropout=dropout,
                        **kwargs,
                    )
                )

            if i > 0:
                if pixel_shuffle:
                    blocks.append(
                        nn.Sequential(
                            ConvNd(
                                hid_channels[i],
                                hid_channels[i - 1] * math.prod(stride),
                                spatial=spatial,
                                identity_init=True,
                                **kwargs,
                            ),
                            Unpatchify(patch_size=stride),
                        )
                    )
                else:
                    blocks.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=tuple(stride), mode="nearest"),
                            ConvNd(
                                hid_channels[i],
                                hid_channels[i - 1],
                                spatial=spatial,
                                identity_init=True,
                                **kwargs,
                            ),
                        )
                    )
            else:
                blocks.append(ConvNd(hid_channels[i], out_channels, spatial=spatial, **kwargs))

            self.ascent.append(blocks)

        self.checkpointing = checkpointing

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor, with shape :math:`(B, C_i, L_1, ..., L_N)`.

        Returns:
            The output tensor, with shape :math:`(B, C_o, L_1 \times 2^D, ..., L_N  \times 2^D)`.
        """

        for blocks in self.ascent:
            for block in blocks:
                if self.checkpointing and isinstance(block, ResBlock):
                    x = checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)

        return x


class AutoEncoder(nn.Module):
    r"""Creates an auto-encoder module.

    Arguments:
        pix_channels: The number of pixel channels :math:`C_i`.
        lat_channels: The number of latent channels :math:`C_o`.
        hid_channels: The numbers of channels at each depth.
        hid_blocks: The numbers of hidden blocks at each depth.
        saturation: The type of latent saturation.
        kwargs: Keyword arguments passed to :class:`Encoder` and :class:`Decoder`.
    """

    def __init__(
        self,
        pix_channels: int,
        lat_channels: int,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        saturation: str = "arcsinh",
        **kwargs,
    ):
        super().__init__()

        self.encoder = Encoder(
            in_channels=pix_channels,
            out_channels=lat_channels,
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            **kwargs,
        )

        self.decoder = Decoder(
            in_channels=lat_channels,
            out_channels=pix_channels,
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            **kwargs,
        )

        self.saturation = saturation

    def saturate(self, x: Tensor) -> Tensor:
        if self.saturation is None:
            return x
        elif self.saturation == "softclip":
            return x / (1 + abs(x) / 5)
        elif self.saturation == "softclip2":
            return x * torch.rsqrt(1 + torch.square(x / 5))
        elif self.saturation == "tanh":
            return torch.tanh(x / 5) * 5
        elif self.saturation == "arcsinh":
            return torch.arcsinh(x)
        else:
            raise ValueError(f"unknown saturation '{self.saturation}'")

    def encode(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        z = self.saturate(z)
        return z

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.encode(x)
        y = self.decoder(z)
        return y, z
