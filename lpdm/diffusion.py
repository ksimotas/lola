r"""Diffusion building blocks."""

__all__ = [
    "ImprovedPreconditionedDenoiser",
    "DenoiserLoss",
    "get_autoencoder",
    "get_denoiser",
]

import math
import torch
import torch.nn as nn

from azula.denoise import Gaussian, GaussianDenoiser, PreconditionedDenoiser
from azula.nn.utils import FlattenWrapper
from azula.noise import Schedule, VESchedule
from omegaconf import DictConfig
from torch import BoolTensor, Tensor
from torch.distributions import Beta, Distribution, Kumaraswamy
from torch.nn.parallel import DistributedDataParallel
from typing import Dict, Optional, Sequence, Tuple, Union

from .nn.autoencoder import AutoEncoder
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


class LogLinearSchedule(Schedule):
    r"""Creates a log-linear noise schedule.

    .. math::
        \alpha_t & = 1 \\
        \sigma_t & = \exp \big( (1 - t) \log \sigma_\min + t \log \sigma_\max \big)

    Arguments:
        sigma_min: The initial noise scale :math:`\sigma_\min \in \mathbb{R}_+`.
        sigma_max: The final noise scale :math:`\sigma_\max \in \mathbb{R}_+`.
    """

    def __init__(self, sigma_min: float = 1e-3, sigma_max: float = 1e2):
        super().__init__()

        self.log_sigma_min = math.log(sigma_min)
        self.log_sigma_max = math.log(sigma_max)

    def alpha(self, t: Tensor) -> Tensor:
        return torch.ones_like(t)

    def sigma(self, t: Tensor) -> Tensor:
        return torch.exp(self.log_sigma_min * (1 - t) + self.log_sigma_max * t)

    def forward(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        return self.alpha(t).unsqueeze(-1), self.sigma(t).unsqueeze(-1)


class LogLogitSchedule(Schedule):
    r"""Creates a log-logit noise schedule.

    .. math::
        \alpha_t & = 1 \\
        \sigma_t & = \exp \log \frac{t}{1 - t} = \frac{t}{1 - t}

    See also:
        :func:`torch.special.logit`

    Arguments:
        sigma_min: The initial noise scale :math:`\sigma_\min \in \mathbb{R}_+`.
    """

    def __init__(self, sigma_min: float = 1e-6):
        super().__init__()

        self.eps = sigma_min / (sigma_min + 1)

    def alpha(self, t: Tensor) -> Tensor:
        return torch.ones_like(t)

    def sigma(self, t: Tensor) -> Tensor:
        return torch.exp(torch.special.logit(t, eps=self.eps))

    def forward(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        return self.alpha(t).unsqueeze(-1), self.sigma(t).unsqueeze(-1)


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


class MaskedDenoiser(GaussianDenoiser):
    r"""Creates a masked denoiser module.

    Arguments:
        denoiser: A Gaussian denoiser.
        y: An observation :math:`y = m \times x`.
        mask: The observation mask :math:`m`.
    """

    def __init__(
        self,
        denoiser: GaussianDenoiser,
        y: Tensor,
        mask: BoolTensor,
    ):
        super().__init__()

        self.denoiser = denoiser

        self.register_buffer("y", torch.as_tensor(y))
        self.register_buffer("mask", torch.as_tensor(mask))

    @property
    def schedule(self) -> Schedule:
        return self.denoiser.schedule

    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> Gaussian:
        alpha_t, sigma_t = self.schedule(t)

        x_t = torch.where(
            self.mask,
            torch.sqrt(alpha_t**2 + sigma_t**2) * self.y,
            x_t,
        )

        return self.denoiser.forward(x_t, t, **kwargs)


class DenoiserLoss(nn.Module):
    r"""Creates a loss module for a Gaussian denoiser."""

    def __init__(self, distribution: str = "beta", a: float = 1.0, b: float = 1.0):
        super().__init__()

        self.distribution = distribution

        self.register_buffer("a", torch.as_tensor(a))
        self.register_buffer("b", torch.as_tensor(b))

    @property
    def prior(self) -> Distribution:
        r"""Returns the time prior :math:`p(t)`."""

        if self.distribution == "beta":
            return Beta(self.a, self.b)
        elif self.distribution == "kumaraswamy":
            return Kumaraswamy(self.a, self.b)
        else:
            raise ValueError(f"unknown distribution {self.distribution}")

    def forward(
        self,
        denoiser: GaussianDenoiser,
        x: Tensor,
        mask: Optional[BoolTensor] = None,
        **kwargs,
    ) -> Tensor:
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

        t = self.prior.sample((B,))

        if isinstance(denoiser, DistributedDataParallel):
            alpha_t, sigma_t = denoiser.module.schedule(t)
        else:
            alpha_t, sigma_t = denoiser.schedule(t)

        x = torch.reshape(x, (B, -1))
        z = torch.randn_like(x)
        x_t = alpha_t * x + sigma_t * z

        if mask is not None:
            x_t = torch.where(
                torch.reshape(mask, (B, -1)),
                torch.sqrt(alpha_t**2 + sigma_t**2) * x,
                x_t,
            )

        q = denoiser(x_t, t, shape=shape, **kwargs)

        l_mean = ((q.mean - x).square() / q.var.detach()).mean()

        if q.var.requires_grad:
            l_var = ((q.mean.detach() - x).square() - q.var).square().mean()
        else:
            l_var = torch.zeros_like(l_mean)

        return l_mean + l_var


def get_autoencoder(
    pix_channels: int,
    lat_channels: int,
    # Common
    hid_channels: Sequence[int],
    hid_blocks: Sequence[int],
    saturation: str = "arcsinh",
    # Future
    arch: Optional[str] = None,
    # Ignore
    name: str = None,
    loss: DictConfig = None,
    # Others
    **kwargs,
) -> AutoEncoder:
    r"""Instantiates an auto-encoder."""

    if arch is None:
        autoencoder = AutoEncoder(
            pix_channels=pix_channels,
            lat_channels=lat_channels,
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            saturation=saturation,
            **kwargs,
        )
    else:
        raise NotImplementedError()

    return autoencoder


def get_denoiser(
    arch: str,
    shape: Sequence[int],
    # Common
    emb_features: int,
    hid_channels: Union[int, Sequence[int]],
    hid_blocks: Union[int, Sequence[int]],
    attention_heads: Union[int, Dict[int, int]],
    dropout: Optional[float] = None,
    # Denoiser
    improved: bool = False,
    schedule: DictConfig = None,
    # DiT
    qk_norm: bool = True,
    patch_size: Union[int, Sequence[int]] = 4,
    window_size: Optional[Sequence[int]] = None,
    rope: bool = True,
    # UNet
    kernel_size: Union[int, Sequence[int]] = 3,
    stride: Union[int, Sequence[int]] = 2,
    norm: str = "layer",
    # Label
    label_features: int = 0,
    # Ignore
    name: str = None,
    loss: DictConfig = None,
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
            spatial=len(shape) - 1,
            patch_size=patch_size,
            window_size=window_size,
            rope=rope,
            dropout=dropout,
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
            norm=norm,
            attention_heads=attention_heads,
            spatial=len(shape) - 1,
            dropout=dropout,
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

    if schedule is None:
        schedule = VESchedule(sigma_min=1e-3, sigma_max=1e3)
    elif schedule.name == "log_linear":
        schedule = LogLinearSchedule(sigma_min=schedule.sigma_min, sigma_max=schedule.sigma_max)
    elif schedule.name == "log_logit":
        schedule = LogLogitSchedule(sigma_min=schedule.sigma_min)

    if improved:
        denoiser = ImprovedPreconditionedDenoiser(backbone, schedule)
    else:
        denoiser = PreconditionedDenoiser(backbone, schedule)

    return denoiser
