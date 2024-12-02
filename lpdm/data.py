r"""Data and datasets helpers."""

__all__ = [
    "field_preprocess",
    "field_postprocess",
    "find_hdf5",
    "get_well_dataset",
    "get_dataloader",
    "infinite_cycle",
    "isotropic_power_spectrum",
    "LazyShuffleDataset",
    "MiniWellDataset",
]

import glob
import h5py
import itertools
import math
import numpy as np
import os
import random
import torch

from the_well.data import WellDataset
from the_well.data.augmentation import (
    Augmentation,
    Compose,
    RandomAxisFlip,
    RandomAxisPermute,
    RandomAxisRoll,
)
from the_well.data.datasets import TrajectoryData, TrajectoryMetadata
from torch import Tensor
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    DistributedSampler,
    IterableDataset,
)
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

TRANSFORMS = {
    "log": (torch.log, torch.exp),
    "log1p": (torch.log1p, torch.expm1),
    "arcsinh": (torch.arcsinh, torch.sinh),
}


def get_label(item: Dict[str, Tensor]):
    r"""Returns the label of an item."""

    scalars = item["constant_scalars"]
    bcs = item["boundary_conditions"]

    *batch, _ = scalars.shape

    return torch.cat((scalars, bcs.reshape(*batch, -1)), dim=-1)


def field_preprocess(
    x: Tensor,
    mean: Optional[Tensor] = None,
    std: Optional[Tensor] = None,
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

    if mean is not None:
        x = x - mean.to(x)

    if std is not None:
        x = x / std.to(x)

    return x


def field_postprocess(
    x: Tensor,
    mean: Optional[Tensor] = None,
    std: Optional[Tensor] = None,
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

    if std is not None:
        x = x * std.to(x)

    if mean is not None:
        x = x + mean.to(x)

    for i, key in transform.items():
        _, t_inv = TRANSFORMS[key]
        x[..., i] = t_inv(x[..., i])

    return x


def find_hdf5(
    path: str,
    include_filters: Sequence[str] = (),
    exclude_filters: Sequence[str] = (),
) -> List[str]:
    r"""Finds HDF5 files in a directory.

    Arguments:
        path: A path to a directory of HDF5 files.
        include_filters: Include files whose name contains any of these strings.
        exclude_filters: Exclude files whose name contains any of these strings

    Returns:
        The list of HDF5 files.
    """

    path = os.path.realpath(os.path.expanduser(path), strict=True)

    files = glob.glob(os.path.join(path, "*.hdf5")) + glob.glob(os.path.join(path, "*.h5"))
    files = sorted(files)

    if include_filters:
        files = [file for file in files if any(filtr in file for filtr in include_filters)]

    if exclude_filters:
        files = [file for file in files if not any(filtr in file for filtr in exclude_filters)]

    return files


class LogScalars(Augmentation):
    def __call__(self, data: TrajectoryData, metadata: TrajectoryMetadata) -> TrajectoryData:
        scalars = data["constant_scalars"]

        for key, value in scalars.items():
            if key.lower() in ("rayleigh", "prandtl"):
                scalars[key] = value.log()

        data["constant_scalars"] = scalars

        return data


def compose_augmentation(*names: str) -> Augmentation:
    augmentations = []

    for name in names:
        if name == "random_axis_flip":
            augmentations.append(RandomAxisFlip())
        elif name == "random_axis_permute":
            augmentations.append(RandomAxisPermute())
        elif name == "random_axis_roll":
            augmentations.append(RandomAxisRoll())
        elif name == "log_scalars":
            augmentations.append(LogScalars())
        else:
            raise NotImplementedError()

    return Compose(*augmentations)


def get_well_multi_dataset(
    path: str,
    physics: Union[str, Sequence[str]],
    **kwargs,
):
    if isinstance(physics, str):
        physics = [physics]

    datasets = [get_well_dataset(path, physic, **kwargs) for physic in physics]

    dataset = ConcatDataset(datasets)
    dataset.metadata = datasets[0].metadata

    return dataset


def get_well_dataset(
    path: str,
    physics: str,
    split: str,
    steps: int = 1,
    include_filters: Sequence[str] = (),
    exclude_filters: Sequence[str] = (),
    augment: Sequence[str] = (),
    **kwargs,
) -> WellDataset:
    r"""Instantiates a dataset from the Well.

    Arguments:
        path: A path to a directory of HDF5 datasets.
        split: The name of the data split to load. Options are "train", "valid", and "test".
            If the path does not contain the split directory, an exception is raised.
        steps: The number of time steps in the trajectories.
        include_filters: Include files whose name contains any of these strings.
        exclude_filters: Exclude files whose name contains any of these strings
        kwargs: Keyword arguments passed to :class:`the_well.data.WellDataset`.

    Returns:
        A dataset from the Well.
    """

    path = os.path.realpath(os.path.expanduser(path), strict=True)

    if os.path.exists(os.path.join(path, physics)):
        path = os.path.join(path, physics)
    elif os.path.exists(os.path.join(path, "datasets", physics)):
        path = os.path.join(path, "datasets", physics)
    else:
        raise NotADirectoryError(f"{os.path.join(path, physics)} does not exist.")

    if os.path.exists(os.path.join(path, split)):
        path = os.path.join(path, split)
    elif os.path.exists(os.path.join(path, "data", split)):
        path = os.path.join(path, "data", split)
    else:
        raise NotADirectoryError(f"{os.path.join(path, split)} does not exist.")

    if augment:
        augmentation = compose_augmentation(*augment)
    else:
        augmentation = None

    return WellDataset(
        path=path,
        n_steps_input=steps,
        n_steps_output=0,
        include_filters=include_filters,
        exclude_filters=exclude_filters,
        use_normalization=False,
        transform=augmentation,
        **kwargs,
    )


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: Union[bool, str] = False,
    infinite: bool = False,
    num_workers: int = 1,
    persistent_workers: bool = True,
    pin_memory: bool = True,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    seed: int = 0,
) -> DataLoader:
    r"""Instantiates a (distributed) data loader."""

    if shuffle == "lazy":
        dataset = LazyShuffleDataset(dataset)
        sampler = None
        shuffle = None
    else:
        if rank is None:
            sampler = None
        else:
            sampler = DistributedSampler(
                dataset=dataset,
                drop_last=True,
                shuffle=shuffle,
                rank=rank,
                num_replicas=world_size,
                seed=seed,
            )

    def worker_init_fn(worker_id: int):
        worker_seed = torch.initial_seed() % 2**32
        worker_seed = worker_seed + rank

        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    loader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=True,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )

    if infinite:
        return infinite_cycle(loader)
    else:
        return loader


def infinite_cycle(loader: DataLoader) -> Iterator[Any]:
    r"""Creates an infinite iterator over a data loader."""

    for epoch in itertools.count():
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)

        yield from loader


def isotropic_power_spectrum(
    x: Tensor,
    edges: Optional[Tensor] = None,
    spatial: int = 2,
) -> Tuple[Tensor, Tensor]:
    r"""Computes the isotropic power spectrum of a field.

    Adapted from :func:`the_well.benchmark.metrics.spectral.power_spectrum`.

    Arguments:
        x: A field tensor, with shape :math:`(*, H_1, ..., H_N)`.
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


class LazyShuffleDataset(IterableDataset):
    r"""Creates a lazily shuffled dataset."""

    def __init__(self, dataset: Dataset, chunk_size: int = 16, buffer_size: int = 256):
        super().__init__()

        self.dataset = dataset
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterable[Any]:
        buffer = []
        chunks = [
            (i, min(i + self.chunk_size, len(self))) for i in range(0, len(self), self.chunk_size)
        ]

        random.shuffle(chunks)

        while True:
            while chunks and len(buffer) < self.buffer_size:
                start, stop = chunks.pop()
                for i in range(start, stop):
                    buffer.append(self.dataset[i])

            if buffer:
                i = random.randrange(len(buffer))
                yield buffer.pop(i)
            else:
                break


class MiniWellDataset(Dataset):
    r"""Creates a mini-Well dataset."""

    def __init__(
        self,
        file: str,
        steps: int = 1,
        stride: int = 1,
    ):
        self.file = h5py.File(file, mode="r")

        self.trajectories = self.file["state"].shape[0]
        self.steps_per_trajectory = self.file["state"].shape[1]

        self.steps = steps
        self.stride = stride

    def __len__(self) -> int:
        return self.trajectories * (self.steps_per_trajectory - (self.steps - 1) * self.stride)

    def __getitem__(self, i: int) -> Dict[str, Tensor]:
        crops_per_trajectory = self.steps_per_trajectory - (self.steps - 1) * self.stride

        i, j = i // crops_per_trajectory, i % crops_per_trajectory

        state = self.file["state"][
            i, slice(j, j + (self.steps - 1) * self.stride + 1, self.stride)
        ]
        label = self.file["label"][i]

        return {
            "state": torch.as_tensor(state),
            "label": torch.as_tensor(label),
        }

    @staticmethod
    def from_files(files: Iterable[str], **kwargs) -> Dataset:
        return ConcatDataset([MiniWellDataset(file, **kwargs) for file in files])
