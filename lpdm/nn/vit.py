r"""Vision Transformer (ViT) building blocks.

References:
    | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Dosovitskiy et al., 2021)
    | https://arxiv.org/abs/2010.11929

    | Scalable Diffusion Models with Transformers (Peebles et al., 2022)
    | https://arxiv.org/abs/2212.09748
"""

__all__ = [
    "ViTBlock",
    "ViT",
]

import functools
import itertools
import math
import torch
import torch.nn as nn
import xformers.components.attention.core as xfa

from einops.layers.torch import Rearrange
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from typing import Hashable, Optional, Sequence, Tuple, Union

from .attention import MultiheadSelfAttention


class ViTBlock(nn.Module):
    r"""Creates a ViT block module.

    Arguments:
        channels: The number of channels :math:`C`.
        mod_features: The number of modulating features :math:`D`.
        spatial: The number of spatial dimensinons :math:`N`.
        rope: Whether to use rotary positional embedding (RoPE) or not.
        dropout: The dropout rate in :math:`[0, 1]`.
        checkpointing: Whether to use gradient checkpointing or not.
        kwargs: Keyword arguments passed to :class:`MultiheadSelfAttention`.
    """

    def __init__(
        self,
        channels: int,
        mod_features: int,
        spatial: int = 2,
        rope: bool = True,
        dropout: Optional[float] = None,
        checkpointing: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.checkpointing = checkpointing

        # Ada-LN Zero
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)
        self.ada_zero = nn.Sequential(
            nn.Linear(mod_features, channels),
            nn.SiLU(),
            nn.Linear(channels, 6 * channels),
            Rearrange("... (n C) -> n ... 1 C", n=6),
        )

        self.ada_zero[-2].weight.data.mul_(1e-2)

        # MSA
        self.msa = MultiheadSelfAttention(channels, **kwargs)

        ## RoPE
        if rope:
            amplitude = 1e2 ** -torch.rand(channels // 2)
            direction = torch.nn.functional.normalize(torch.randn(spatial, channels // 2), dim=0)

            self.theta = nn.Parameter(amplitude * direction)
        else:
            self.theta = None

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.SiLU(),
            nn.Identity() if dropout is None else nn.Dropout(dropout),
            nn.Linear(4 * channels, channels),
        )

    def _forward(
        self,
        x: Tensor,
        mod: Tensor,
        indices: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Arguments:
            x: The input tokens :math:`x`, with shape :math:`(*, L, C)`.
            mod: The modulation vector, with shape :math:`(D)` or :math:`(*, D)`.
            indices: The postition indices, with shape :math:`(*, L, N)`.
            mask: The attention mask, with shape :math:`(*, L, L)`.

        Returns:
            The ouput tokens :math:`y`, with shape :math:`(*, L, C)`.
        """

        if indices is None or self.theta is None:
            theta = None
        else:
            theta = torch.einsum("...ij,jk", indices, self.theta)

        a1, b1, c1, a2, b2, c2 = self.ada_zero(mod)

        y = (a1 + 1) * self.norm(x) + b1
        y = self.msa(y, theta, mask)
        y = (x + c1 * y) * torch.rsqrt(1 + c1 * c1)

        y = (a2 + 1) * self.norm(y) + b2
        y = self.mlp(y)
        y = (x + c2 * y) * torch.rsqrt(1 + c2 * c2)

        return y

    def forward(
        self,
        x: Tensor,
        mod: Tensor,
        indices: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.checkpointing:
            return checkpoint(self._forward, x, mod, indices, mask, use_reentrant=False)
        else:
            return self._forward(x, mod, indices, mask)


class ViT(nn.Module):
    r"""Creates a modulated ViT-like module.

    Arguments:
        in_channels: The number of input channels :math:`C_i`.
        out_channels: The number of output channels :math:`C_o`.
        cond_channels: The number of condition channels :math:`C_c`.
        mod_features: The number of modulating features :math:`D`.
        hid_channels: The numbers of hidden token channels.
        hid_blocks: The number of hidden transformer blocks.
        attention_heads: The number of attention heads :math:`H`.
        spatial: The number of spatial dimensions :math:`N`.
        patch_size: The path size.
        window_size: The local attention window size.
        rope: Whether to use rotary positional embedding (RoPE) or not.
        dropout: The dropout rate in :math:`[0, 1]`.
        kwargs: Keyword arguments passed to :class:`ViTBlock`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int = 0,
        mod_features: int = 0,
        hid_channels: int = 1024,
        hid_blocks: int = 3,
        attention_heads: int = 1,
        spatial: int = 2,
        patch_size: Union[int, Sequence[int]] = 4,
        window_size: Union[int, Sequence[int], None] = None,
        rope: bool = True,
        dropout: Optional[float] = None,
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

        self.in_proj = nn.Linear(
            math.prod(patch_size) * (in_channels + cond_channels), hid_channels
        )
        self.out_proj = nn.Linear(hid_channels, math.prod(patch_size) * out_channels)

        self.positional_embedding = nn.Sequential(
            nn.Linear(spatial, hid_channels),
            nn.SiLU(),
            nn.Linear(hid_channels, hid_channels),
        )

        self.blocks = nn.ModuleList([
            ViTBlock(
                channels=hid_channels,
                mod_features=mod_features,
                spatial=spatial,
                **kwargs,
            )
            for _ in range(hid_blocks)
        ])

        self.spatial = spatial

        if window_size is None:
            self.window_size = None
        elif isinstance(window_size, int):
            self.window_size = (window_size,) * spatial
        else:
            self.window_size = tuple(window_size)

    @staticmethod
    @functools.cache
    def indices_and_mask(
        shape: Sequence[int],
        spatial: int,
        window_size: Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        r"""Returns the token indices and attention mask for a given input shape and window size."""

        assert isinstance(shape, Hashable)
        assert isinstance(window_size, Hashable)

        indices = (torch.arange(size, device=device) for size in shape)
        indices = torch.cartesian_prod(*indices)
        indices = torch.reshape(indices, shape=(-1, spatial))

        if window_size is None:
            mask = None
        else:
            delta = torch.abs(indices[:, None] - indices[None, :])
            delta = torch.minimum(delta, delta.new_tensor(shape) - delta)

            mask = torch.all(delta <= indices.new_tensor(window_size) // 2, dim=-1)

            if xfa._has_cpp_library:
                mask = xfa.SparseCS(mask, device=mask.device)._mat

        return indices.to(dtype=dtype), mask

    def forward(
        self,
        x: Tensor,
        mod: Tensor,
        cond: Optional[Tensor] = None,
        early_out: Optional[int] = None,
    ) -> Tensor:
        r"""
        Arguments:
            x: The input tensor, with shape :math:`(B, C_i, L_1, ..., L_N)`.
            mod: The modulation vector, with shape :math:`(D)` or :math:`(B, D)`.
            cond: The condition tensor, with :math:`(B, C_c, L_1, ..., L_N)`.
            early_out: The number of blocks after which the output is returned.

        Returns:
            The output tensor, with shape :math:`(B, C_o, L_1, ..., L_N)`.
        """

        if cond is not None:
            x = torch.cat((x, cond), dim=1)

        x = self.patch(x)
        x = self.in_proj(x)

        shape = x.shape[-self.spatial - 1 : -1]
        indices, mask = self.indices_and_mask(
            shape,
            spatial=self.spatial,
            window_size=self.window_size,
            dtype=x.dtype,
            device=x.device,
        )

        x = torch.flatten(x, -self.spatial - 1, -2)
        x = x + self.positional_embedding(indices / indices.new_tensor(shape))

        for block in itertools.islice(self.blocks, early_out):
            x = block(x, mod, indices=indices, mask=mask)

        x = torch.unflatten(x, sizes=shape, dim=-2)

        x = self.out_proj(x)
        x = self.unpatch(x)

        return x
