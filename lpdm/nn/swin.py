r"""Swin transformer building blocks.

References:
    | Swin Transformer: Hierarchical Vision Transformer using Shifted Windows (Liu et al., 2021)
    | https://arxiv.org/abs/2103.14030
"""

__all__ = [
    "SwinBlock",
    "Swin",
]

import functools
import itertools
import math
import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor
from typing import Hashable, Optional, Sequence, Union

from .attention import MultiheadSelfAttention
from .layers import LayerNorm


class SwinBlock(nn.Module):
    r"""Creates a Swin transformer block module.

    Arguments:
        channels: The number of channels :math:`C`.
        mod_features: The number of modulating features :math:`D`.
        spatial: The number of spatial dimensions :math:`N`.
        window_size: The local attention window size.
        shifted: Whether the windows are shifted or not.
        rope: Whether to use rotary positional embedding (RoPE) or not.
        kwargs: Keyword arguments passed to :class:`MultiheadSelfAttention`.
    """

    def __init__(
        self,
        channels: int,
        mod_features: int,
        spatial: int = 2,
        window_size: Sequence[int] = (4, 4),
        shifted: bool = False,
        rope: bool = True,
        **kwargs,
    ):
        super().__init__()

        assert len(window_size) == spatial

        # Ada-zero
        self.norm = LayerNorm(dim=-1)
        self.ada_zero = nn.Sequential(
            nn.Linear(mod_features, mod_features),
            nn.SiLU(),
            nn.Linear(mod_features, 6 * channels),
            Rearrange("... (n C) -> n ..." + " 1" * spatial + " C", n=6),
        )

        layer = self.ada_zero[-2]
        layer.weight = nn.Parameter(layer.weight * 1e-2)

        # Window
        self.spatial = spatial
        self.window_size = window_size
        self.shifted = shifted

        # MSA
        self.msa = MultiheadSelfAttention(channels, **kwargs)

        ## RoPE
        if rope:
            amplitude = 1e4 ** -torch.rand(spatial, channels // 2)
            direction = torch.nn.functional.normalize(torch.randn(spatial, channels // 2), dim=0)

            self.theta = nn.Parameter(amplitude * direction)
        else:
            self.theta = None

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.SiLU(),
            nn.Linear(4 * channels, channels),
        )

    def shift(self, x: Tensor) -> Tensor:
        if self.shifted:
            return torch.roll(
                x,
                shifts=[0 if w is None else w // 2 for w in self.window_size],
                dims=list(range(-self.spatial - 1, -1)),
            )
        else:
            return x

    def unshift(self, x: Tensor) -> Tensor:
        if self.shifted:
            return torch.roll(
                x,
                shifts=[0 if w is None else -(w // 2) for w in self.window_size],
                dims=list(range(-self.spatial - 1, -1)),
            )
        else:
            return x

    def group(self, x: Tensor, shape: Sequence[int]) -> Tensor:
        window_size = [s if w is None else w for s, w in zip(shape, self.window_size, strict=True)]

        if self.spatial == 1:
            (l,) = window_size
            return rearrange(x, "... (L l) C -> ... L (l) C", l=l)
        elif self.spatial == 2:
            h, w = window_size
            return rearrange(x, "... (H h) (W w) C -> ... H W (h w) C", h=h, w=w)
        elif self.spatial == 3:
            l, h, w = window_size
            return rearrange(x, "... (L l) (H h) (W w) C -> ... L H W (l h w) C", l=l, h=h, w=w)
        else:
            raise NotImplementedError()

    def ungroup(self, x: Tensor, shape: Sequence[int]) -> Tensor:
        window_size = [s if w is None else w for s, w in zip(shape, self.window_size, strict=True)]

        if self.spatial == 1:
            (l,) = window_size
            return rearrange(x, "... L (l) C -> ... (L l) C", l=l)
        elif self.spatial == 2:
            h, w = window_size
            return rearrange(x, "... H W (h w) C -> ... (H h) (W w) C", h=h, w=w)
        elif self.spatial == 3:
            l, h, w = window_size
            return rearrange(x, "... L H W (l h w) C -> ... (L l) (H h) (W w) C", l=l, h=h, w=w)
        else:
            raise NotImplementedError()

    @staticmethod
    @functools.cache
    def indices(
        shape: Sequence[int],
        spatial: int,
        window_size: Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        r"""Returns the token indices for a given input shape and window size."""

        assert isinstance(shape, Hashable)
        assert isinstance(window_size, Hashable)

        indices = [
            torch.arange(s if w is None else w, dtype=dtype, device=device)
            for s, w in zip(shape, window_size, strict=True)
        ]
        indices = torch.cartesian_prod(*indices)
        indices = torch.reshape(indices, shape=(-1, spatial))

        return indices

    def forward(self, x: Tensor, mod: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tokens :math:`x`, with shape :math:`(*, H_1, ..., H_N, C)`.
            mod: The modulation vector, with shape :math:`(D)` or :math:`(*, D)`.

        Returns:
            The ouput tokens :math:`y`, with shape :math:`(*, H_1, ..., H_N, C)`.
        """

        shape = x.shape[-self.spatial - 1 : -1]

        if self.theta is None:
            theta = None
        else:
            indices = self.indices(
                shape,
                spatial=self.spatial,
                window_size=tuple(self.window_size),
                dtype=x.dtype,
                device=x.device,
            )

            theta = torch.einsum("...ij,jk", indices, self.theta)

        a1, b1, c1, a2, b2, c2 = self.ada_zero(mod)

        y = (a1 + 1) * self.norm(x) + b1
        y = self.shift(y)
        y = self.group(y, shape)
        y = self.msa(y, theta)
        y = self.ungroup(y, shape)
        y = self.unshift(y)
        y = (x + c1 * y) * torch.rsqrt(1 + c1 * c1)

        y = (a2 + 1) * self.norm(y) + b2
        y = self.mlp(y)
        y = (x + c2 * y) * torch.rsqrt(1 + c2 * c2)

        return y


class Swin(nn.Module):
    r"""Creates a modulated non-hierarchical Swin-like module.

    Arguments:
        in_channels: The number of input channels :math:`C_i`.
        out_channels: The number of output channels :math:`C_o`.
        mod_features: The number of modulating features :math:`D`.
        hid_channels: The numbers of hidden token channels.
        hid_blocks: The number of hidden transformer blocks.
        attention_heads: The number of attention heads :math:`H`.
        dropout: The dropout rate in :math:`[0, 1]`.
        spatial: The number of spatial dimensions :math:`N`.
        patch_size: The path size.
        window_size: The local attention window size.
        rope: Whether to use rotary positional embedding (RoPE) or not.
        kwargs: Keyword arguments passed to :class:`SwinBlock`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mod_features: int,
        hid_channels: int = 1024,
        hid_blocks: int = 3,
        attention_heads: int = 1,
        dropout: Optional[float] = None,
        spatial: int = 2,
        patch_size: Union[int, Sequence[int]] = 4,
        window_size: Sequence[int] = (4, 4),
        rope: bool = True,
        **kwargs,
    ):
        super().__init__()

        kwargs.update(
            attention_heads=attention_heads,
            dropout=dropout,
            rope=rope,
        )

        if isinstance(patch_size, int):
            patch_size = [patch_size] * spatial

        if spatial == 1:
            (l,) = patch_size
            self.patch = Rearrange("... C (L l) -> ... L (C l)", l=l)
            self.unpatch = Rearrange("... L (C l) -> ... C (L l)", l=l)
        elif spatial == 2:
            h, w = patch_size
            self.patch = Rearrange("... C (H h) (W w) -> ... H W (C h w)", h=h, w=w)
            self.unpatch = Rearrange("... H W (C h w) -> ... C (H h) (W w)", h=h, w=w)
        elif spatial == 3:
            l, h, w = patch_size
            self.patch = Rearrange("... C (L l) (H h) (W w) -> ... L H W (C l h w)", l=l, h=h, w=w)
            self.unpatch = Rearrange(
                "... L H W (C l h w) -> ... C (L l) (H h) (W w)", l=l, h=h, w=w
            )
        else:
            raise NotImplementedError()

        self.in_proj = nn.Linear(math.prod(patch_size) * in_channels, hid_channels)
        self.out_proj = nn.Linear(hid_channels, math.prod(patch_size) * out_channels)

        self.positional_embedding = nn.Sequential(
            nn.Linear(spatial, hid_channels),
            nn.SiLU(),
            nn.Linear(hid_channels, hid_channels),
        )

        self.blocks = nn.ModuleList([
            SwinBlock(
                channels=hid_channels,
                mod_features=mod_features,
                spatial=spatial,
                window_size=window_size,
                shifted=i % 2 == 0,
                **kwargs,
            )
            for i in range(hid_blocks)
        ])

        self.spatial = spatial

    @staticmethod
    @functools.cache
    def indices(
        shape: Sequence[int],
        spatial: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        r"""Returns the token indices for a given input shape."""

        assert isinstance(shape, Hashable)

        indices = [torch.arange(size, dtype=dtype, device=device) for size in shape]
        indices = torch.cartesian_prod(*indices)
        indices = torch.reshape(indices, shape=(*shape, spatial))

        return indices / indices.new_tensor(shape)

    def forward(self, x: Tensor, mod: Tensor, early_out: Optional[int] = None) -> Tensor:
        r"""
        Arguments:
            x: The input tensor, with shape :math:`(B, C_i, H_1, ..., H_N)`.
            mod: The modulation vector, with shape :math:`(D)` or :math:`(B, D)`.
            early_out: The number of blocks after which the output is returned.

        Returns:
            The output tensor, with shape :math:`(B, C_o, H_1, ..., H_N)`.
        """

        x = self.patch(x)
        x = self.in_proj(x)

        shape = x.shape[-self.spatial - 1 : -1]
        indices = self.indices(
            shape,
            spatial=self.spatial,
            dtype=x.dtype,
            device=x.device,
        )

        x = x + self.positional_embedding(indices)

        for block in itertools.islice(self.blocks, early_out):
            x = block(x, mod)

        x = self.out_proj(x)
        x = self.unpatch(x)

        return x
