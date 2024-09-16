r"""Data and datasets helpers."""

__all__ = [
    "field_preprocess",
    "field_postprocess",
    "get_well_dataset",
    "isotropic_power_spectrum",
]

import glob
import math
import os
import shutil
import torch

from torch import Tensor
from typing import Dict, Optional, Sequence, Tuple

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
    in_steps: int = 1,
    out_steps: int = 0,
    include_filters: Sequence[str] = (),
    exclude_filters: Sequence[str] = (),
    memory_mapped: bool = False,
    shm: str = "/dev/shm",
    **kwargs,
) -> GenericWellDataset:
    r"""Instantiates a dataset from the Well.

    Arguments:
        path: A path to a directory of HDF5 datasets.
        in_steps: The number of time steps in the input trajectories.
        out_steps: The number of time steps in the output trajectories.
        include_filters: Include files whose name contains any of these strings.
        exclude_filters: Exclude files whose name contains any of these strings
        memory_mapped: Whether to map files to memory or not.
        shm: The shared memory filesystem.
        kwargs: Keyword arguments passed to :class:`GenericWellDataset`.

    Returns:
        A dataset from the Well.
    """

    path = os.path.abspath(os.path.expanduser(path))

    if memory_mapped:
        files = glob.glob(os.path.join(path, "*.h5"))
        files += glob.glob(os.path.join(path, "*.hdf5"))

        if include_filters:
            files = [file for file in files if any(filtr in file for filtr in include_filters)]

        if exclude_filters:
            files = [file for file in files if all(filtr not in file for filtr in exclude_filters)]

        for file in files:
            map_to_memory(file, shm=shm)

        path = os.path.join(shm, os.path.relpath(path, "/"))

    return GenericWellDataset(
        path=path,
        n_steps_input=in_steps,
        n_steps_output=out_steps,
        include_filters=include_filters,
        exclude_filters=exclude_filters,
        use_normalization=False,
        **kwargs,
    )


def isotropic_power_spectrum(
    x: Tensor,
    edges: Optional[Tensor] = None,
    spatial: int = 2,
) -> Tuple[Tensor, Tensor]:
    r"""Computes the isotropic power spectrum of a field.

    Adapted from :func:`the_well.benchmark.metrics.spectral.power_spectrum`.

    Arguments:
        x: A field tensor, with shape :math:`(*, H_1, ..., H_N,)`.
        edges: The wavelength bin edges, with shape :math:`(L + 1)`.
        spatial: The number of spatial dimensions :math:`N`.

    Returns:
        The power spectrum and the wavelength bin edges, with shape :math:`(*, L)` and
        :math:`(L + 1)`, respectively.
    """

    # Wavelength
    w2_iso = 0

    for i in range(spatial):
        w_i = torch.fft.fftfreq(x.shape[-i - 1], dtype=x.dtype, device=x.device)
        w_i = w_i.reshape([1] * (spatial - i - 1) + [-1] + [1] * i)
        w2_iso = w2_iso + w_i**2

    w_iso = torch.sqrt(w2_iso)

    if edges is None:
        bins = math.ceil(math.sqrt(max(w_iso.shape)))
        edges = torch.linspace(0, w_iso.max(), bins + 1, dtype=x.dtype, device=x.device)
    else:
        bins = len(edges) - 1

    indices = torch.clip(torch.bucketize(w_iso.flatten(), edges), max=bins - 1)
    counts = torch.bincount(indices, minlength=bins)

    # Power spectrum
    p = torch.fft.fftn(x, dim=tuple(range(-spatial, 0))).abs().square()
    p = torch.flatten(p, -spatial)

    p_iso = torch.zeros((*p.shape[:-1], bins), dtype=x.dtype, device=x.device)
    p_iso = p_iso.scatter_add(dim=-1, index=indices.expand_as(p), src=p)
    p_iso = p_iso / torch.clip(counts, min=1)

    return p_iso, edges


def map_to_memory(
    file: str,
    shm: str = "/dev/shm",
    exist_ok: bool = False,
) -> str:
    r"""Maps a file to memory.

    Arguments:
        file: The source file to map.
        shm: The shared memory filesystem.

    Returns:
        The file's destination.
    """

    src = os.path.abspath(os.path.expanduser(file))
    dst = os.path.join(shm, os.path.relpath(file, "/"))

    if os.path.exists(dst):
        if exist_ok:
            return dst
        else:
            raise FileExistsError(f"{dst} already exists.")
    else:
        os.makedirs(os.path.dirname(dst), exist_ok=True)

    size = os.path.getsize(src)
    free = os.statvfs(shm).f_frsize * os.statvfs(shm).f_bavail

    if size < free:
        return shutil.copy2(src, dst)
    else:
        raise MemoryError(f"not enough space on {shm} (needed: {size} B, free: {free} B).")
