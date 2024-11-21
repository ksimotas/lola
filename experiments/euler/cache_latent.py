#!/usr/bin/env python

import argparse
import dawgz
import glob
import os

from typing import Sequence


def cache_latent(
    autoencoder: str,
    physics: str,
    split: str,
    file: str,
    augment: Sequence[str] = (),
    duplicate: int = 1,
    datasets: str = "~/ceph/the_well/datasets",
):
    import h5py
    import numpy as np
    import torch

    from einops import rearrange
    from functools import partial
    from omegaconf import OmegaConf
    from pathlib import Path
    from tqdm import trange

    from lpdm.data import field_preprocess, get_well_dataset
    from lpdm.diffusion import get_autoencoder

    device = torch.device("cuda")

    # Config
    runpath = Path(autoencoder)
    runpath = runpath.expanduser().resolve(strict=True)

    cfg = OmegaConf.load(runpath / "config.yaml")

    cache_path = runpath / "cache" / physics / split / file

    # Data
    dataset = get_well_dataset(
        path=datasets,
        physics=physics,
        split=split,
        include_filters=[file],
    )

    dataset = get_well_dataset(
        path=datasets,
        physics=physics,
        split=split,
        steps=dataset.metadata.n_steps_per_trajectory[0],
        include_filters=[file],
        augment=augment,
    )

    preprocess = partial(
        field_preprocess,
        mean=torch.as_tensor(cfg.dataset.stats.mean, device=device),
        std=torch.as_tensor(cfg.dataset.stats.std, device=device),
        transform=cfg.dataset.transform,
    )

    # Autoencoder
    autoencoder = get_autoencoder(
        pix_channels=dataset.metadata.n_fields,
        **cfg.ae,
    )

    state = torch.load(runpath / "state.pth", weights_only=True)

    if "predictor" in state:
        state = state["autoencoder"]

    autoencoder.load_state_dict(state)
    autoencoder.to(device)
    autoencoder.eval()

    # Encode
    if cache_path.exists():
        return
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(cache_path, mode="x") as f:
        for i in trange(len(dataset), ncols=88, ascii=True):
            visited = []

            for j in range(duplicate):
                while True:
                    batch = dataset[i]

                    x = batch["input_fields"]
                    x = x.to(device, non_blocking=True)
                    x = preprocess(x)
                    x = rearrange(x, "L H W C -> L C H W")

                    for y in visited:
                        if torch.allclose(x, y):
                            break
                    else:
                        break

                visited.append(x)

                with torch.no_grad():
                    z = autoencoder.encode(x)

                label = torch.cat([
                    batch["constant_scalars"].reshape(-1),
                    batch["boundary_conditions"].reshape(-1),
                ])

                if "state" not in f:
                    f.create_dataset(
                        "state",
                        shape=(len(dataset) * duplicate, *z.shape),
                        dtype=np.float32,
                    )

                if "label" not in f:
                    f.create_dataset(
                        "label",
                        shape=(len(dataset) * duplicate, *label.shape),
                        dtype=np.float32,
                    )

                f["state"][i * duplicate + j] = z.numpy(force=True)
                f["label"][i * duplicate + j] = label.numpy(force=True)


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("autoencoder", type=str)
    parser.add_argument("--physics", default="euler_multi_quadrants_openBC", type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--filters", nargs="*", type=str)
    parser.add_argument("--augment", nargs="*", type=str)
    parser.add_argument("--duplicate", default=1, type=int)

    args = parser.parse_args()

    # Files
    datasets = os.path.expanduser("~/ceph/the_well/datasets")
    path = os.path.join(datasets, args.physics, "data", args.split)

    files = glob.glob("*.hdf5", root_dir=path) + glob.glob("*.h5", root_dir=path)

    if args.filters:
        files = [file for file in files if any(filtr in file for filtr in args.filters)]

    # Job(s)
    def launch(i: int):
        cache_latent(
            autoencoder=args.autoencoder,
            physics=args.physics,
            split=args.split,
            file=files[i],
            augment=args.augment,
            duplicate=args.duplicate,
        )

    dawgz.schedule(
        dawgz.job(
            f=launch,
            name="cache_latent",
            array=len(files),
            cpus=4,
            gpus=1,
            ram="64GB",
            time="06:00:00",
            partition="gpu",
            constraint="h100|a100",
        ),
        name=f"caching {args.physics}/{args.split}",
        backend="slurm",
        env=[
            "export XDG_CACHE_HOME=$HOME/.cache",
        ],
    )
