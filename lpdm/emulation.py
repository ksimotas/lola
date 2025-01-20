r"""Physics emulation helpers."""

import torch

from azula.sample import DDIMSampler, DDPMSampler, LMSSampler, PCSampler
from einops import rearrange
from torch import BoolTensor, Tensor
from typing import Callable, Optional

from .diffusion import GaussianDenoiser, MaskedDenoiser
from .nn.autoencoder import AutoEncoder
from .surrogate import MaskedSurrogate


def encode_traj(autoencoder: AutoEncoder, x: Tensor) -> Tensor:
    x = rearrange(x, "C L ... -> L C ...")
    z = autoencoder.encode(x)
    z = rearrange(z, "L C ... -> C L ...")

    return z


def decode_traj(autoencoder: AutoEncoder, z: Tensor) -> Tensor:
    z = rearrange(z, "C L ... -> L C ...")
    x = autoencoder.decode(z)
    x = rearrange(x, "L C ... -> C L ...")

    return x


def emulate_surrogate(
    surrogate: MaskedSurrogate,
    mask: BoolTensor,
    x_obs: Tensor,
    label: Optional[Tensor] = None,
) -> Tensor:
    mask = mask.unsqueeze(dim=0)

    y = torch.zeros(mask.shape, dtype=x_obs.dtype, device=x_obs.device)
    y[mask] = x_obs.flatten()

    x_hat = surrogate(y, mask=mask, label=label)
    x_hat = x_hat.squeeze(dim=0)

    return x_hat


def emulate_diffusion(
    denoiser: GaussianDenoiser,
    mask: BoolTensor,
    x_obs: Tensor,
    label: Optional[Tensor] = None,
    algorithm: str = "lms",
    steps: int = 64,
    **kwargs,
) -> Tensor:
    y = torch.zeros(mask.shape, dtype=x_obs.dtype, device=x_obs.device)
    y[mask] = x_obs.flatten()

    cond_denoiser = MaskedDenoiser(
        denoiser,
        y=y.flatten(),
        mask=mask.flatten(),
    )

    if algorithm == "ddpm":
        cond_sampler = DDPMSampler(cond_denoiser, steps=steps, **kwargs)
    elif algorithm == "ddim":
        cond_sampler = DDIMSampler(cond_denoiser, steps=steps, **kwargs)
    elif algorithm == "lms":
        cond_sampler = LMSSampler(cond_denoiser, steps=steps, **kwargs)
    elif algorithm == "pc":
        cond_sampler = PCSampler(cond_denoiser, steps=steps, **kwargs)

    cond_sampler = cond_sampler.to(x_obs.device)

    x1 = cond_sampler.init((1, mask.numel()))
    x0 = cond_sampler(x1, label=label, cond=mask.unsqueeze(dim=0))
    x0 = x0.reshape(mask.shape)

    return x0


def emulate_rollout(
    emulate: Callable[[BoolTensor, Tensor], Tensor],
    x: Tensor,
    window: int,
    rollout: int,
    context: int = 1,
    overlap: int = 1,
) -> Tensor:
    assert context > 0
    assert overlap > 0
    assert window > context and window > overlap

    x_obs = x[:, :context]

    mask = torch.zeros_like(x[:, :window], dtype=bool)
    mask[:, :context] = True

    trajectory = []

    while len(trajectory) < rollout:
        x_hat = emulate(mask, x_obs)

        if trajectory:
            trajectory.extend(x_hat[:, overlap:].unbind(dim=1))
        else:
            trajectory.extend(x_hat.unbind(dim=1))

        x_obs = x_hat[:, -overlap:]

        mask = torch.zeros_like(mask)
        mask[:, :overlap] = True

    trajectory = trajectory[:rollout]

    return torch.stack(trajectory, dim=1)


def random_context_mask(
    x: Tensor,
    lmbda: float = 1.0,
    rho: float = 0.5,
    atleast: int = 0,
) -> BoolTensor:
    r"""Returns a random context mask.

    The task is either to emulate forward or backward in time. Consequently, the context
    is a contiguous chunk of states at the beginning or end of the trajectory with
    probability :math:`\rho`.

    Arguments:
        x: A trajectory tensor, with shape :math:`(B, C, L, ...)`.
        lmbda: The average number of states :math:`\lambda` in the context. The number
            of states in the context is at most :math:`L - 1`.
        rho: The probability :math:`\rho` for the context to be at the beginning.
        atleast: The minimum number of context states. :math:`\lambda` does not take
            this number into account.
    """

    B, _, L, *shape = x.shape

    rate = torch.full((B, 1), fill_value=lmbda, device=x.device)
    context = torch.poisson(rate).long()
    context = context % (L - atleast) + atleast

    index = torch.arange(L, device=x.device)
    forward = torch.rand((B, 1), device=x.device) < rho

    mask = torch.where(
        forward,
        index < context,
        index >= L - context,
    )

    mask = mask.reshape([B, 1, L] + [1 for _ in shape])
    mask = mask.expand_as(x)

    return mask
