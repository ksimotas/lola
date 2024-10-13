#!/usr/bin/env python

import argparse
import dawgz
import glob
import os


def cache_latent(
    autoencoder: str,
    physics: str,
    split: str,
    file: str,
    datasets: str = "/mnt/ceph/users/polymathic/the_well/datasets",
):
    import h5py
    import numpy as np
    import os
    import torch

    from einops import rearrange
    from functools import partial
    from omegaconf import OmegaConf
    from pathlib import Path
    from tqdm import trange

    from lpdm.data import field_preprocess, get_well_dataset
    from lpdm.nn.autoencoder import AutoEncoder

    device = torch.device("cuda")

    # Config
    runpath = Path(autoencoder)
    runpath = runpath.expanduser().resolve(strict=True)

    cfg = OmegaConf.load(runpath / "config.yaml")

    cache_path = runpath / "cache" / physics / split / file

    if cache_path.exists():
        return
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Data
    dataset = get_well_dataset(
        path=os.path.join(datasets, physics),
        split=split,
        include_filters=[file],
    )

    dataset = get_well_dataset(
        path=os.path.join(datasets, physics),
        split=split,
        steps=dataset.metadata.n_steps_per_simulation[0],
        include_filters=[file],
    )

    preprocess = partial(
        field_preprocess,
        mean=torch.as_tensor(cfg.dataset.stats.mean, device=device),
        std=torch.as_tensor(cfg.dataset.stats.std, device=device),
        transform=cfg.dataset.transform,
    )

    # Autoencoder
    autoencoder = AutoEncoder(
        pix_channels=dataset.metadata.n_fields,
        **cfg.ae,
    )

    autoencoder.load_state_dict(torch.load(runpath / "state.pth"))
    autoencoder.to(device)
    autoencoder.eval()

    # Encode
    with h5py.File(cache_path, mode="x") as f:
        for i in trange(len(dataset), ncols=88, ascii=True):
            batch = dataset[i]

            x = batch["input_fields"]
            x = x.to(device, non_blocking=True)
            x = preprocess(x)
            x = rearrange(x, "L H W C -> L C H W")

            with torch.no_grad():
                z = autoencoder.encode(x)

            label = batch["constant_scalars"]

            if "state" not in f:
                f.create_dataset("state", shape=(len(dataset), *z.shape), dtype=np.float32)

            if "label" not in f:
                f.create_dataset("label", shape=(len(dataset), *label.shape), dtype=np.float32)

            f["state"][i] = z.numpy(force=True)
            f["label"][i] = label.numpy(force=True)


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("autoencoder", type=str)
    parser.add_argument("--physics", default="euler_multi_quadrants_openBC", type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--filters", nargs="*", type=str)

    args = parser.parse_args()

    # Files
    datasets = "/mnt/ceph/users/polymathic/the_well/datasets"
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
        )

    dawgz.schedule(
        dawgz.job(
            f=launch,
            name="cache_latent",
            array=len(files),
            cpus=4,
            gpus=1,
            ram="64GB",
            time="01:00:00",
            partition="gpu",
            constraint="h100",
        ),
        name=f"caching {args.physics}/{args.split}",
        backend="slurm",
        env=[
            "export XDG_CACHE_HOME=$HOME/.cache",
        ],
    )
