r"""Optimization and training helpers."""

__all__ = [
    "get_optimizer",
    "safe_gd_step",
]

import math
import torch
import torch.nn as nn

from torch import Tensor
from typing import Iterable, Optional, Tuple


def get_optimizer(
    params: Iterable[nn.Parameter],
    optimizer: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    scheduler: Optional[str] = None,
    epochs: Optional[int] = None,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    r"""Instantiates an optimizer and sheduler.

    Arguments:
        params: The optimized parameters.
        optimizer: The optimizer name.
        learning_rate: The learning rate.
        weight_decay: The weight decay.
        scheduler: The scheduler name.
        epochs: The total number of epochs.

    Returns:
        An optimizer/scheduler pair.
    """

    if optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params=params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        raise NotImplementedError()

    if scheduler is None:
        lr_lambda = lambda t: 1
    elif scheduler == "linear":
        lr_lambda = lambda t: max(0, 1 - (t / epochs))
    elif scheduler == "cosine":
        lr_lambda = lambda t: (1 + math.cos(math.pi * t / epochs)) / 2
    elif scheduler == "exponential":
        lr_lambda = lambda t: math.exp(math.log(1e-6) * t / epochs)
    else:
        raise NotImplementedError()

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def safe_gd_step(
    loss: Tensor,
    optimizer: torch.optim.Optimizer,
    grad_clip: Optional[float] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Applies a gradient descent (GD) optimization step.

    To prevent invalid parameters, steps are skipped if not-a-number (NaN) or infinite
    values are found in the gradient. This feature requires CPU-GPU synchronization,
    which could be a bottleneck for some applications.

    Arguments:
        loss: The loss value, with shape :math:`()`.
        optimizer: An optimizer.
        grad_clip: The maximum gradient norm. If :py:`None`, gradients are not clipped.

    Returns:
        The detached loss and the unclipped gradient norm.
    """

    norm = None

    if loss.isfinite():
        loss.backward()

        if grad_clip is None:
            optimizer.step()
            optimizer.zero_grad()
        else:
            params = (p for group in optimizer.param_groups for p in group["params"])
            norm = nn.utils.clip_grad_norm_(params, grad_clip)

            if norm.isfinite():
                optimizer.step()
                optimizer.zero_grad()

    return loss.detach(), norm
