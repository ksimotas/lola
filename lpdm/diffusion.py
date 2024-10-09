r"""Diffusion building blocks."""

__all__ = [
    "ImprovedPreconditionedDenoiser",
    "DenoiserLoss",
    "get_denoiser",
]

import math
import torch
import torch.nn as nn

from azula.denoise import Gaussian, GaussianDenoiser, PreconditionedDenoiser
from azula.nn.utils import FlattenWrapper
from azula.noise import Schedule, VESchedule
from torch import Tensor
from torch.distributions import Beta
from typing import Dict, Optional, Sequence, Union

# isort: split
from .nn.dit import DiT
from .nn.embedding import SineEncoding
from .nn.unet import UNet


class EmbeddingWrapper(nn.Module):
    r"""Creates a time/label embedding wrapper around a modulated backbone."""

    def __init__(
        self,
        backbone: nn.Module,
        time_embedding: nn.Module,
        label_embedding: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.backbone = backbone
        self.time_embedding = time_embedding
        self.label_embedding = label_embedding

    def forward(self, x_t: Tensor, t: Tensor, label: Optional[Tensor] = None, **kwargs):
        if label is None:
            emb = self.time_embedding(t)
        else:
            emb = self.time_embedding(t) + self.label_embedding(label)

        return self.backbone(x_t, emb, **kwargs)


class ImprovedPreconditionedDenoiser(GaussianDenoiser):
    r"""Creates an improved preconditioned denoiser.

    Arguments:
        backbone: A noise conditional network.
        schedule: A noise schedule.
    """

    def __init__(self, backbone: nn.Module, schedule: Schedule):
        super().__init__()

        self.backbone = backbone
        self.schedule = schedule

    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> Gaussian:
        alpha_t, sigma_t = self.schedule(t)

        c_in = 1 / torch.sqrt(alpha_t**2 + sigma_t**2)
        c_out = sigma_t / torch.sqrt(alpha_t**2 + sigma_t**2)
        c_skip = alpha_t / (alpha_t**2 + sigma_t**2)
        c_noise = torch.log(sigma_t / alpha_t).squeeze(dim=-1)

        output = self.backbone(c_in * x_t, c_noise, **kwargs)
        mean, log_var = torch.chunk(output, chunks=2, dim=-1)

        mean = c_skip * x_t + c_out * mean
        var = sigma_t**2 / (alpha_t**2 + sigma_t**2) * torch.exp(log_var)

        return Gaussian(mean=mean, var=var)


class DenoiserLoss(nn.Module):
    r"""Creates a loss module for a Gaussian denoiser."""

    def __init__(self, denoiser: GaussianDenoiser, a: float = 1.0, b: float = 1.0):
        super().__init__()

        self.denoiser = denoiser

        self.register_buffer("a", torch.as_tensor(a))
        self.register_buffer("b", torch.as_tensor(b))

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        r"""
        Arguments:
            x: A clean vector :math:`x`, with shape :math:`(B, ...)`.
            kwargs: Optional keyword arguments.

        Returns:
            The (averaged) negative log-likelihood

            .. math:: -\log \mathcal{N}(x \mid \mu_\phi(x_t), \Sigma_\phi(x_t))

            where :math:`t \sim Beta(a, b)` and :math:`x_t \sim p(X_t \mid x)`.
        """

        B, *shape = x.shape

        x = torch.reshape(x, (B, -1))
        t = Beta(self.a, self.b).sample((B,))

        nll = self.denoiser.loss(x, t, shape=shape, **kwargs)

        return nll.mean() / math.prod(shape)


def get_denoiser(
    arch: str,
    shape: Sequence[int],
    emb_features: int,
    hid_channels: Union[int, Sequence[int]],
    hid_blocks: Union[int, Sequence[int]],
    attention_heads: Union[int, Dict[int, int]],
    dropout: float = 0.1,
    # Denoiser
    improved: bool = True,
    # DiT
    qk_norm: bool = True,
    patch_size: Union[int, Sequence[int]] = 4,
    rope: bool = True,
    registers: int = 0,
    # UNet
    kernel_size: Union[int, Sequence[int]] = 3,
    stride: Union[int, Sequence[int]] = 2,
    # Label
    label_features: int = 0,
    # Ignore
    name: str = None,
) -> GaussianDenoiser:
    r"""Instantiates a denoiser."""

    channels, *_ = shape

    if arch == "dit":
        backbone = DiT(
            in_channels=channels,
            out_channels=2 * channels if improved else channels,
            mod_features=emb_features,
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            attention_heads=attention_heads,
            qk_norm=qk_norm,
            dropout=dropout,
            spatial=len(shape) - 1,
            patch_size=patch_size,
            rope=rope,
            registers=registers,
        )
    elif arch == "unet":
        backbone = UNet(
            in_channels=channels,
            out_channels=2 * channels if improved else channels,
            mod_features=emb_features,
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            kernel_size=kernel_size,
            stride=stride,
            attention_heads=attention_heads,
            dropout=dropout,
            spatial=len(shape) - 1,
        )
    else:
        raise NotImplementedError()

    time_embedding = SineEncoding(emb_features)

    if label_features > 0:
        label_embedding = nn.Sequential(
            nn.Linear(label_features, emb_features),
            nn.SiLU(),
            nn.Linear(emb_features, emb_features),
        )
    else:
        label_embedding = None

    backbone = FlattenWrapper(
        wrappee=EmbeddingWrapper(
            backbone=backbone,
            time_embedding=time_embedding,
            label_embedding=label_embedding,
        ),
        shape=shape,
    )

    schedule = VESchedule(sigma_min=1e-3, sigma_max=1e3)

    if improved:
        denoiser = ImprovedPreconditionedDenoiser(backbone, schedule)
    else:
        denoiser = PreconditionedDenoiser(backbone, schedule)

    return denoiser
