r"""Auto-encoder building blocks."""

__all__ = [
    "AutoEncoder",
    "AutoEncoderLoss",
    "get_autoencoder",
]

import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange
from omegaconf import DictConfig
from torch import Tensor
from torch.nn.functional import cosine_similarity
from typing import Any, Dict, Optional, Sequence, Tuple

from .nn.dcae import DCDecoder, DCEncoder
from .nn.vit import ViT


class AutoEncoder(nn.Module):
    r"""Creates an auto-encoder module.

    Arguments:
        encoder: An encoder module.
        decoder: A decoder module.
        saturation: The type of latent saturation.
        noise: The latent noise's standard deviation.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        saturation: str = "softclip2",
        noise: float = 0.0,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.saturation = saturation
        self.noise = noise

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
        if self.noise > 0:
            z = z + self.noise * torch.randn_like(z)

        return self.decoder(z)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.encode(x)
        y = self.decoder(z)
        return y, z


class AutoEncoderLoss(nn.Module):
    r"""Creates a weighted auto-encoder loss module."""

    def __init__(
        self,
        losses: Sequence[str] = ["mse"],  # noqa: B006
        weights: Sequence[float] = [1.0],  # noqa: B006
    ):
        super().__init__()

        assert len(losses) == len(weights)

        self.losses = list(losses)
        self.register_buffer("weights", torch.as_tensor(weights))

    def forward(self, autoencoder: AutoEncoder, x: Tensor, **kwargs) -> Tensor:
        r"""
        Arguments:
            x: A clean tensor :math:`x`, with shape :math:`(B, ...)`.
            kwargs: Optional keyword arguments.

        Returns:
            The weighted loss.
        """

        y, z = autoencoder(x, **kwargs)

        values = []

        for loss in self.losses:
            if loss == "mse":
                l = (x - y).square().mean()
            elif loss == "mae":
                l = (x - y).abs().mean()
            elif loss == "similarity":
                f = rearrange(z, "B ... -> B (...)")
                l = cosine_similarity(f[None, :], f[:, None], dim=-1)
                l = l[*torch.triu_indices(*l.shape, offset=1, device=l.device)]
                l = l.mean()
            else:
                raise ValueError(f"unknown loss '{loss}'.")

            values.append(l)

        values = torch.stack(values)

        return torch.vdot(self.weights, values)


def get_autoencoder(
    pix_channels: int,
    lat_channels: int,
    spatial: int = 2,
    # Arch
    arch: Optional[str] = None,
    saturation: str = "softclip2",
    # Asymmetry
    encoder_only: Dict[str, Any] = {},  # noqa: B006
    decoder_only: Dict[str, Any] = {},  # noqa: B006
    # Noise
    latent_noise: float = 0.0,
    # Ignore
    name: str = None,
    loss: DictConfig = None,
    # Passthrough
    **kwargs,
) -> nn.Module:
    r"""Instantiates an auto-encoder."""

    if arch in (None, "dcae"):
        encoder = DCEncoder(
            in_channels=pix_channels,
            out_channels=lat_channels,
            spatial=spatial,
            **encoder_only,
            **kwargs,
        )

        decoder = DCDecoder(
            in_channels=lat_channels,
            out_channels=pix_channels,
            spatial=spatial,
            **decoder_only,
            **kwargs,
        )
    elif arch == "vit":
        encoder = ViT(
            in_channels=pix_channels,
            out_channels=lat_channels,
            spatial=spatial,
            **encoder_only,
            **kwargs,
        )

        encoder.out_proj = nn.Linear(encoder.out_proj.in_features, lat_channels)
        encoder.unpatch = Rearrange("B ... C -> B C ...")

        decoder = ViT(
            in_channels=lat_channels,
            out_channels=pix_channels,
            spatial=spatial,
            **decoder_only,
            **kwargs,
        )

        decoder.in_proj = nn.Linear(lat_channels, decoder.in_proj.out_features)
        decoder.patch = Rearrange("B C ... -> B ... C")
    else:
        raise NotImplementedError()

    autoencoder = AutoEncoder(
        encoder,
        decoder,
        saturation=saturation,
        noise=latent_noise,
    )

    return autoencoder
