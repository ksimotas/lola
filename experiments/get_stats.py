#!/usr/bin/env python

import argparse
import dawgz

from functools import partial
from omegaconf import DictConfig, OmegaConf

from lpdm.hydra import compose


def get_stats(
    dataset: DictConfig,
    samples: int,
    datasets: str = "/mnt/ceph/users/polymathic/the_well/datasets",
):
    import itertools
    import torch

    from einops import rearrange
    from tabulate import tabulate
    from torch.utils.data import DataLoader

    from lpdm.data import field_preprocess, get_well_multi_dataset
    from lpdm.utils import process_cpu_count

    print(OmegaConf.to_yaml(dataset), flush=True)
    print()

    # Data
    trainset = get_well_multi_dataset(
        path=datasets,
        physics=dataset.physics,
        split="train",
        steps=1,
        include_filters=dataset.include_filters,
    )

    train_loader = DataLoader(
        dataset=trainset,
        batch_size=None,
        shuffle=True,
        num_workers=process_cpu_count(),
    )

    preprocess = partial(
        field_preprocess,
        transform=dataset.transform,
    )

    # Fetch
    mins = []
    maxs = []
    first_moments = []
    second_moments = []

    for batch in itertools.islice(train_loader, samples):
        x = batch["input_fields"]
        x = preprocess(x)
        x = rearrange(x, "... C -> (...) C")

        mins.append(x.min(dim=0).values)
        maxs.append(x.max(dim=0).values)
        first_moments.append(x.mean(dim=0))
        second_moments.append(x.square().mean(dim=0))

    mins = torch.stack(mins).min(dim=0).values
    maxs = torch.stack(maxs).max(dim=0).values
    first_moments = torch.stack(first_moments).mean(dim=0)
    second_moments = torch.stack(second_moments).mean(dim=0)

    stats = {
        "min": mins.tolist(),
        "max": maxs.tolist(),
        "mean": first_moments.tolist(),
        "std": torch.sqrt(second_moments - first_moments**2).tolist(),
    }

    rows = [(k, *v) for k, v in stats.items()]

    print(tabulate(rows, headers=("Stat", *dataset.fields)), flush=True)


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./configs/dataset/euler_open.yaml", type=str)
    parser.add_argument("--samples", type=int, default=1024)

    args = parser.parse_args()

    # Config
    dataset = compose(config_file=args.dataset)

    # Job
    dawgz.schedule(
        dawgz.job(
            f=partial(get_stats, dataset, args.samples),
            name="get_stats",
            cpus=16,
            gpus=1,
            ram="64GB",
            time="00:15:00",
            partition="gpu",
        ),
        name=f"stats {dataset.name}",
        backend="slurm",
    )
