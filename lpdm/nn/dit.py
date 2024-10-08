r"""Diffusion Transformer (DiT) building blocks."""

__all__ = [
    "DiTBlock",
    "DiT",
]

import math
import torch
import torch.nn as nn
import xformers.components.attention.core as xf

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from typing import Optional, Sequence, Tuple, Union

# isort: split
from .common import LayerNorm, RMSNorm


class MultiheadSelfAttention(nn.Module):
    r"""Creates a multi-head self-attention layer.

    Arguments:
        channels: The number of channels :math:`H \times C`.
        attention_heads: The number of attention heads :math:`H`.
        dropout: The dropout rate in :math:`[0, 1]`.
        qk_norm: Whether to use query-key RMS-normalization or not.
        checkpointing: Whether to use gradient checkpointing or not.
    """

    def __init__(
        self,
        channels: int,
        attention_heads: int = 1,
        dropout: float = 0.0,
        qk_norm: bool = True,
        checkpointing: bool = True,
    ):
        super().__init__()

        assert channels % attention_heads == 0

        self.qkv_proj = nn.Linear(channels, 3 * channels, bias=False)
        self.y_proj = nn.Linear(channels, channels)

        if qk_norm:
            self.qk_norm = RMSNorm(dim=-1)
        else:
            self.qk_norm = nn.Identity()

        self.heads = attention_heads
        self.dropout = nn.Dropout(dropout)
        self.checkpointing = checkpointing

    def _forward(
        self,
        x: Tensor,
        theta: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Arguments:
            x: The input tokens :math:`x`, with shape :math:`(*, L, H \times C)`.
            theta: Optional rotary positional embedding :math:`\theta`,
                with shape :math:`(*, L, H \times C / 2)`.

        Returns:
            The ouput tokens :math:`y`, with shape :math:`(*, L, H \times C)`.
        """

        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, "... L (n H C) -> n ... H L C", n=3, H=self.heads)
        q, k = self.qk_norm(q), self.qk_norm(k)

        if theta is not None:
            theta = rearrange(theta, "... L (H C) -> ... H L C", H=self.heads)
            q, k = self.apply_rope(q, k, theta)

        if xf._has_cpp_library and isinstance(mask, xf.SparseCS):
            y = xf.scaled_dot_product_attention(
                q=rearrange(q, "B H L C -> (B H) L C"),
                k=rearrange(k, "B H L C -> (B H) L C"),
                v=rearrange(v, "B H L C -> (B H) L C"),
                att_mask=mask,
                dropout=self.dropout if self.training else None,
            )
            y = rearrange(y, "(B H) L C -> B H L C", H=self.heads)
        else:
            y = torch.nn.functional.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0,
            )

        y = rearrange(y, "... H L C -> ... L (H C)")
        y = self.y_proj(y)

        return y

    @staticmethod
    def apply_rope(q: Tensor, k: Tensor, theta: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        References:
            | RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)
            | https://arxiv.org/abs/2104.09864

            | Rotary Position Embedding for Vision Transformer (Heo et al., 2024)
            | https://arxiv.org/abs/2403.13298

        Arguments:
            q: The query tokens :math:`q`, with shape :math:`(*, L, C)`.
            k: The key tokens :math:`k`, with shape :math:`(*, L, C)`.
            theta: Rotary angles, with shape :math:`(*, L, C / 2)`.

        Returns:
            The rotated query and key tokens, with shape :math:`(*, L, C)`.
        """

        rotation = torch.polar(torch.ones_like(theta), theta)

        q = torch.view_as_complex(torch.unflatten(q, -1, (-1, 2)))
        k = torch.view_as_complex(torch.unflatten(k, -1, (-1, 2)))

        q = torch.flatten(torch.view_as_real(rotation * q), -2)
        k = torch.flatten(torch.view_as_real(rotation * k), -2)

        return q, k

    def forward(
        self,
        x: Tensor,
        theta: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.checkpointing:
            return checkpoint(self._forward, x, theta, mask, use_reentrant=False)
        else:
            return self._forward(x, theta, mask)


class DiTBlock(nn.Module):
    r"""Creates a DiT block module.

    Arguments:
        channels: The number of channels :math:`C`.
        mod_features: The number of modulating features :math:`D`.
        spatial: The number of spatial dimensinons :math:`N` for RoPE.
        kwargs: Keyword arguments passed to :class:`MultiheadSelfAttention`.
    """

    def __init__(
        self,
        channels: int,
        mod_features: int,
        spatial: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        # Ada-zero
        self.norm = LayerNorm(dim=-1)
        self.ada_zero = nn.Sequential(
            nn.Linear(mod_features, mod_features),
            nn.SiLU(),
            nn.Linear(mod_features, 6 * channels),
            Rearrange("... (n C) -> n ... 1 C", n=6),
        )

        layer = self.ada_zero[-2]
        layer.weight = nn.Parameter(layer.weight * 1e-2)

        # MSA
        self.msa = MultiheadSelfAttention(channels, **kwargs)

        ## RoPE
        if spatial is None:
            self.theta = None
        else:
            amplitude = 1e4 ** -torch.rand(channels // 2)
            direction = torch.nn.functional.normalize(torch.randn(spatial, channels // 2), dim=0)

            self.theta = nn.Parameter(amplitude * direction)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.SiLU(),
            nn.Linear(4 * channels, channels),
        )

    def forward(
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


class DiT(nn.Module):
    r"""Creates a modulated DiT-like module.

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
        registers: The number of registers.
        kwargs: Keyword arguments passed to :class:`DiTBlock`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mod_features: int,
        hid_channels: int = 1024,
        hid_blocks: int = 3,
        attention_heads: int = 1,
        dropout: float = 0.0,
        spatial: int = 2,
        patch_size: Union[int, Sequence[int]] = 4,
        window_size: Optional[Sequence[int]] = None,
        rope: bool = True,
        registers: int = 0,
        **kwargs,
    ):
        super().__init__()

        kwargs.update(
            attention_heads=attention_heads,
            dropout=dropout,
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
            DiTBlock(
                channels=hid_channels,
                mod_features=mod_features,
                spatial=spatial if rope else None,
                **kwargs,
            )
            for _ in range(hid_blocks)
        ])

        self.spatial = spatial
        self.window_size = window_size

        self.register_tokens = nn.Parameter(torch.randn(registers, hid_channels))
        self.register_buffer(
            "register_indices",
            repeat(torch.arange(-registers, 0), "R -> R n", n=spatial).clone(),
        )

    def forward(self, x: Tensor, mod: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor, with shape :math:`(B, C_i, H_1, ..., H_N)`.
            mod: The modulation vector, with shape :math:`(D)` or :math:`(B, D)`.

        Returns:
            The output tensor, with shape :math:`(B, C_o, H_1, ..., H_N)`.
        """

        x = self.patch(x)
        x = self.in_proj(x)

        shape = x.shape[-self.spatial - 1 : -1]
        numel = math.prod(shape)

        indices = (torch.arange(size, device=x.device) for size in shape)
        indices = torch.cartesian_prod(*indices)
        indices = torch.reshape(indices, shape=(-1, len(shape)))
        indices = torch.cat((indices, self.register_indices), dim=-2)

        if self.window_size is None:
            mask = None
        else:
            delta = torch.abs(indices[:, None] - indices[None, :])
            delta = torch.minimum(delta, delta.new_tensor(shape) - delta)

            mask = torch.all(delta <= indices.new_tensor(self.window_size), dim=-1)
            mask[numel:, :] = True
            mask[:, numel:] = True

            if xf._has_cpp_library:
                mask = xf.SparseCS(mask, device=mask.device)

        indices = indices.to(dtype=x.dtype)

        x = torch.flatten(x, -self.spatial - 1, -2)
        x = torch.cat((x, self.register_tokens.expand(*x.shape[:-2], -1, -1)), dim=-2)

        x = x + self.positional_embedding(indices / indices.new_tensor(shape))

        for block in self.blocks:
            x = block(x, mod, indices=indices, mask=mask)

        x = torch.narrow(x, start=0, length=numel, dim=-2)
        x = torch.unflatten(x, sizes=shape, dim=-2)

        x = self.out_proj(x)
        x = self.unpatch(x)

        return x
