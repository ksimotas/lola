r"""Auto-encoder building blocks."""

__all__ = [
    "ResBlock",
    "Encoder",
    "Decoder",
    "AutoEncoder",
]

import torch
import torch.nn as nn

from torch import Tensor
from typing import Dict, Optional, Sequence, Union

# isort: split
from .common import (
    ConvNd,
    LayerNorm,
    SelfAttentionNd,
    SpectralConvNd,
    ViewAsComplex,
    ViewAsReal,
)


class Residual(nn.Sequential):
    def forward(self, x: Tensor) -> Tensor:
        return x + super().forward(x)


class ResBlock(nn.Module):
    r"""Creates a residual block module.

    Arguments:
        channels: The number of channels :math:`C`.
        attention_heads: The number of attention heads.
        spectral_modes: The number of spectral convolution modes.
        dropout: The dropout rate in :math:`[0, 1]`.
        spatial: The number of spatial dimensions :math:`N`.
        kwargs: Keyword arguments passed to :class:`torch.nn.Conv2d`.
    """

    def __init__(
        self,
        channels: int,
        attention_heads: Optional[int] = None,
        spectral_modes: Optional[int] = None,
        dropout: Optional[float] = None,
        spatial: int = 2,
        **kwargs,
    ):
        super().__init__()

        # Ada-zero
        self.ada_zero = nn.Parameter(1e-2 * torch.randn((3, channels) + (1,) * spatial))

        # Block
        self.block = nn.Sequential(
            LayerNorm(dim=1),
            ConvNd(channels, channels, spatial=spatial, **kwargs),
            nn.SiLU(),
            nn.Identity() if dropout is None else nn.Dropout(dropout),
            ConvNd(channels, channels, spatial=spatial, **kwargs),
        )

        if attention_heads is not None:
            self.block.append(
                Residual(
                    LayerNorm(dim=1),
                    SelfAttentionNd(channels, heads=attention_heads),
                )
            )

        if spectral_modes is not None:
            self.block.append(
                Residual(
                    ViewAsComplex(dim=1),
                    SpectralConvNd(
                        channels // 2,
                        channels // 2,
                        modes=spectral_modes,
                        spatial=spatial,
                    ),
                    ViewAsReal(dim=1),
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor, with shape :math:`(B, C, H_1, ..., H_N)`.

        Returns:
            The output tensor, with shape :math:`(B, C, H_1, ..., H_N)`.
        """

        a, b, c = self.ada_zero

        y = (a + 1) * x + b
        y = self.block(y)
        y = x + c * y
        y = y / torch.sqrt(1 + c * c)

        return y


class Encoder(nn.Module):
    r"""Creates an encoder module.

    Arguments:
        in_channels: The number of input channels :math:`C_i`.
        out_channels: The number of output channels :math:`C_o`.
        hid_channels: The numbers of channels at each depth.
        hid_blocks: The numbers of hidden blocks at each depth.
        kernel_size: The kernel size of all convolutions.
        stride: The stride of the downsampling convolutions.
        attention_heads: The number of attention heads at each depth.
        spectral_modes: The number of spectral convolution modes at each depth.
        dropout: The dropout rate in :math:`[0, 1]`.
        spatial: The number of spatial dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 2,
        attention_heads: Dict[int, int] = {},  # noqa: B006
        spectral_modes: Dict[int, int] = {},  # noqa: B006
        dropout: Optional[float] = None,
        spatial: int = 2,
    ):
        super().__init__()

        assert len(hid_blocks) == len(hid_channels)

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * spatial

        if isinstance(stride, int):
            stride = [stride] * spatial

        kwargs = dict(
            kernel_size=tuple(kernel_size),
            padding=tuple(k // 2 for k in kernel_size),
        )

        self.descent = nn.ModuleList()

        for i, num_blocks in enumerate(hid_blocks):
            blocks = nn.ModuleList()

            if i > 0:
                blocks.append(
                    nn.Sequential(
                        ConvNd(
                            hid_channels[i - 1],
                            hid_channels[i],
                            spatial=spatial,
                            stride=stride,
                            **kwargs,
                        ),
                        LayerNorm(dim=1),
                    ),
                )
            else:
                blocks.append(ConvNd(in_channels, hid_channels[i], spatial=spatial, **kwargs))

            for _ in range(num_blocks):
                blocks.append(
                    ResBlock(
                        hid_channels[i],
                        attention_heads=attention_heads.get(i, None),
                        spectral_modes=spectral_modes.get(i, None),
                        dropout=dropout,
                        spatial=spatial,
                        **kwargs,
                    )
                )

            if i + 1 == len(hid_blocks):
                blocks.append(ConvNd(hid_channels[i], out_channels, spatial=spatial, **kwargs))

            self.descent.append(blocks)

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor, with shape :math:`(B, C_i, H_1, ..., H_N)`.

        Returns:
            The output tensor, with shape :math:`(B, C_o, H_1 / 2^D, ..., H_N  / 2^D)`.
        """

        for blocks in self.descent:
            for block in blocks:
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
        attention_heads: The number of attention heads at each depth.
        spectral_modes: The number of spectral convolution modes at each depth.
        dropout: The dropout rate in :math:`[0, 1]`.
        spatial: The number of spatial dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_channels: Sequence[int] = (256, 128, 64),
        hid_blocks: Sequence[int] = (3, 3, 3),
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 2,
        attention_heads: Dict[int, int] = {},  # noqa: B006
        spectral_modes: Dict[int, int] = {},  # noqa: B006
        dropout: Optional[float] = None,
        spatial: int = 2,
    ):
        super().__init__()

        assert len(hid_blocks) == len(hid_channels)

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * spatial

        if isinstance(stride, int):
            stride = [stride] * spatial

        kwargs = dict(
            kernel_size=tuple(kernel_size),
            padding=tuple(k // 2 for k in kernel_size),
        )

        self.ascent = nn.ModuleList()

        for i, num_blocks in enumerate(hid_blocks):
            blocks = nn.ModuleList()

            if i == 0:
                blocks.append(ConvNd(in_channels, hid_channels[i], spatial=spatial, **kwargs))

            for _ in range(num_blocks):
                blocks.append(
                    ResBlock(
                        hid_channels[i],
                        attention_heads=attention_heads.get(i, None),
                        spectral_modes=spectral_modes.get(i, None),
                        dropout=dropout,
                        spatial=spatial,
                        **kwargs,
                    )
                )

            if i + 1 < len(hid_blocks):
                blocks.append(
                    nn.Sequential(
                        LayerNorm(dim=1),
                        nn.Upsample(scale_factor=tuple(stride), mode="nearest"),
                        ConvNd(
                            hid_channels[i],
                            hid_channels[i + 1],
                            spatial=spatial,
                            **kwargs,
                        ),
                    )
                )
            else:
                blocks.append(ConvNd(hid_channels[i], out_channels, spatial=spatial, **kwargs))

            self.ascent.append(blocks)

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor, with shape :math:`(B, C_i, H_1, ..., H_N)`.

        Returns:
            The output tensor, with shape :math:`(B, C_o, H_1 \times 2^D, ..., H_N  \times 2^D)`.
        """

        for blocks in self.ascent:
            for block in blocks:
                x = block(x)

        return x


class AutoEncoder(nn.Module):
    r"""Creates an auto-encoder module.

    Arguments:
        pix_channels: The number of pixel channels :math:`C_i`.
        lat_channels: The number of latent channels :math:`C_o`.
        hid_channels: The numbers of channels at each depth.
        hid_blocks: The numbers of hidden blocks at each depth.
        kwargs: Keyword arguments passed to :class:`Encoder` and :class:`Decoder`.
    """

    def __init__(
        self,
        pix_channels: int,
        lat_channels: int,
        hid_channels: Sequence[int] = (256, 128, 64),
        hid_blocks: Sequence[int] = (3, 3, 3),
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
            hid_channels=list(reversed(hid_channels)),
            hid_blocks=list(reversed(hid_blocks)),
            **kwargs,
        )

    def saturate(self, x: Tensor) -> Tensor:
        return x / (1 + abs(x) / 5)

    def encode(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        z = self.saturate(z)
        return z

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def loss(self, x: Tensor) -> Tensor:
        y = self.decode(self.encode(x))
        return (x - y).square().mean()
