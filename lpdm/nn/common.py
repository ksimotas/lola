r"""Common layers and modules."""

__all__ = [
    "ConvNd",
    "LayerNorm",
    "SelfAttentionNd",
    "SpectralConvNd",
    "ViewAsComplex",
    "ViewAsReal",
]

import math
import torch
import torch.nn as nn

from einops import rearrange
from torch import Tensor
from typing import Sequence, Union


def ConvNd(in_channels: int, out_channels: int, spatial: int = 2, **kwargs) -> nn.Module:
    r"""Returns an N-dimensional convolutional layer.

    Arguments:
        in_channels: Number of input channels :math:`C_i`.
        out_channels: Number of output channels :math:`C_o`.
        spatial: The number of spatial dimensions :math:`N`.
        kwargs: Keyword arguments passed to :class:`torch.nn.Conv2d`.
    """

    CONVS = {
        1: nn.Conv1d,
        2: nn.Conv2d,
        3: nn.Conv3d,
    }

    if spatial in CONVS:
        Conv = CONVS[spatial]
    else:
        raise NotImplementedError()

    return Conv(in_channels, out_channels, **kwargs)


class LayerNorm(nn.Module):
    r"""Creates a layer that standardizes features along a dimension.

    .. math:: y = \frac{x - \mathbb{E}[x]}{\sqrt{\mathbb{V}[x] + \epsilon}}

    References:
       | Layer Normalization (Lei Ba et al., 2016)
       | https://arxiv.org/abs/1607.06450

    Arguments:
        dim: The dimension(s) to standardize.
        eps: A numerical stability term.
    """

    def __init__(self, dim: Union[int, Sequence[int]], eps: float = 1e-5):
        super().__init__()

        self.dim = dim if isinstance(dim, int) else tuple(dim)

        self.register_buffer("eps", torch.as_tensor(eps))

    def extra_repr(self) -> str:
        return f"dim={self.dim}"

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor :math:`x`, with shape :math:(*).

        Returns:
            The standardized tensor :math:`y`, with shape :math:`(*)`.
        """

        variance, mean = torch.var_mean(x, dim=self.dim, keepdim=True)

        return (x - mean) / (variance + self.eps).sqrt()


class SelfAttentionNd(nn.MultiheadAttention):
    r"""Creates an N-dimensional self-attention layer.

    Arguments:
        channels: The number of channels :math:`C`.
        heads: The number of attention heads.
        kwargs: Keyword arguments passed to :class:`torch.nn.MultiheadAttention`.
    """

    def __init__(self, channels: int, heads: int = 1, **kwargs):
        super().__init__(embed_dim=channels, num_heads=heads, batch_first=True, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor :math:`x`, with shape :math:`(B, C, H_1, ..., H_N)`.

        Returns:
            The ouput tensor :math:`y`, with shape :math:`(B, C, H_1, ..., H_N)`.
        """

        y = rearrange(x, "B C ...  -> B (...) C")
        y, _ = super().forward(y, y, y, average_attn_weights=False)
        y = rearrange(y, "B L C -> B C L").reshape(x.shape)

        return y


class SpectralConvNd(nn.Module):
    r"""Creates a spectral convolution layer.

    .. math:: y = \mathcal{F}^{-1}(W \mathcal{F}(x) + b)

    where :math:`\mathcal{F}` is the discrete Fourier transform.

    Arguments:
        in_channels: The number of input channels :math:`C_i`.
        out_channels: The number of output channels :math:`C_o`.
        modes: The number of spectral modes of the kernel :math:`W` per spatial dimension.
            High(er) frequencies are set to zero.
        bias: Whether the layer learns an additive bias :math:`b` or not.
        spatial: The number of spatial dimensions :math:`N`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: Union[int, Sequence[int]] = 16,
        bias: bool = False,
        spatial: int = 2,
    ):
        super().__init__()

        if isinstance(modes, int):
            modes = [modes] * spatial

        kernel_size = [2 * m for m in modes]

        self.kernel = nn.Parameter(
            torch.randn(*kernel_size, in_channels, out_channels, 2) / math.sqrt(2 * in_channels)
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(*kernel_size, out_channels, 2))
        else:
            self.bias = None

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = tuple(modes)
        self.spatial = spatial

    def extra_repr(self) -> str:
        return f"modes={self.modes}"

    def __call__(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor :math:`x`, with shape :math:`(*, C_i, H_1, \dots, H_n)`.
                Floating point tensors are promoted to complex tensors.

        Returns:
            The output tensor :math:`y`, with shape :math:`(*, C_o, H_1, \dots, H_n)`.
        """

        index = (..., *(slice(2 * m) for m in self.modes), slice(None))

        x = torch.fft.fftn(x, dim=self.spatial_dims, norm="ortho")
        x = torch.roll(x, shifts=tuple(m for m in self.modes), dims=self.spatial_dims)
        x = torch.movedim(x, self.channel_dim, -1)

        y = torch.zeros(*x.shape[:-1], self.out_channels, dtype=x.dtype, device=x.device)

        if self.bias is None:
            y[index] = torch.einsum(
                "...i,...ij->...j",
                x[index],
                torch.view_as_complex(self.kernel),
            )
        else:
            y[index] = torch.einsum(
                "...i,...ij->...j",
                x[index],
                torch.view_as_complex(self.kernel),
            ) + torch.view_as_complex(self.bias)

        y = torch.movedim(y, -1, self.channel_dim)
        y = torch.roll(y, shifts=tuple(-m for m in self.modes), dims=self.spatial_dims)
        y = torch.fft.ifftn(y, dim=self.spatial_dims, norm="ortho")

        return y

    @property
    def channel_dim(self) -> int:
        return -self.spatial - 1

    @property
    def spatial_dims(self) -> Sequence[int]:
        return tuple(range(-self.spatial, 0))


class ViewAsComplex(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return torch.complex(*torch.chunk(x, chunks=2, dim=self.dim))


class ViewAsReal(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat((x.real, x.imag), dim=self.dim)
