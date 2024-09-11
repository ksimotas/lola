r"""Data and datasets helpers."""

import os
import torch

from torch import Tensor
from typing import Dict, Optional

try:
    from the_well.benchmark.data.datasets import GenericWellDataset
except ImportError:
    pass


TRANSFORMS = {
    "log": (torch.log, torch.exp),
}


def field_preprocess(
    x: Tensor,
    mean: Tensor,
    std: Tensor,
    transform: Dict[int, str] = {},  # noqa: B006
) -> Tensor:
    r"""Pre-processes the physical fields of a state.

    Arguments:
        x: A state tensor, with shape :math:`(*, C)`.
        mean: The fields mean, with shape :math:`(C)`.
        std: The field standard deviation, with shape :math:`(C)`.
        transform: Optional (non-linear) transformations to apply to each field.

    Returns:
        The pre-processed state tensor, with shape :math:`(*, C)`.
    """

    for i, key in transform.items():
        t, _ = TRANSFORMS[key]
        x[..., i] = t(x[..., i])

    return (x - mean.to(x)) / std.to(x)


def field_postprocess(
    x: Tensor,
    mean: Tensor,
    std: Tensor,
    transform: Dict[int, str] = {},  # noqa: B006
) -> Tensor:
    r"""Post-processes the physical fields of a state.

    Reverts the pre-processing of :func:`field_preprocess`.

    Arguments:
        x: A pre-processed state tensor, with shape :math:`(*, C)`.
        mean: The fields mean, with shape :math:`(C)`.
        std: The field standard deviation, with shape :math:`(C)`.
        transform: Optional (non-linear) transformations to apply to each field.

    Returns:
        The post-processed state tensor, with shape :math:`(*, C)`.
    """

    x = x * std.to(x) + mean.to(x)

    for i, key in transform.items():
        _, t_inv = TRANSFORMS[key]
        x[..., i] = t_inv(x[..., i])

    return x


def get_well_dataset(
    path: str,
    split: Optional[str] = None,
    in_steps: int = 1,
    out_steps: int = 0,
    **kwargs,
) -> GenericWellDataset:
    r"""Instantiates a dataset from the Well.

    Arguments:
        path: A path to a directory of HDF5 datasets.
        split: The name of the data split to load. Options are "train", "valid", or "test".
            If the path does not contain the split directory, an exception is raised.
        in_steps: The number of time steps in the input trajectories.
        out_steps: The number of time steps in the output trajectories.
        kwargs: Keyword arguments passed to :class:`GenericWellDataset`.

    Returns:
        A dataset from the Well.
    """

    if split is not None:
        if os.path.exists(os.path.join(path, split)):
            path = os.path.join(path, split)
        elif os.path.exists(os.path.join(path, "data", split)):
            path = os.path.join(path, "data", split)
        else:
            raise NotADirectoryError(f"{os.path.join(path, split)} does not exist.")

    return GenericWellDataset(
        path=path,
        n_steps_input=in_steps,
        n_steps_output=out_steps,
        use_normalization=False,
        **kwargs,
    )
