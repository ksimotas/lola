#!/usr/bin/env python

import argparse
import dawgz
import glob
import os

from pathlib import Path
from typing import Sequence

from lpdm.hydra import compose


def cache_latent(
    run: str,
    physics: str,
    split: str,
    file: str,
    augment: Sequence[str] = (),
    repeat: int = 1,
    datasets: str = "/mnt/ceph/users/polymathic/the_well/datasets",
):
    import h5py
    import numpy as np
    import torch

    from einops import rearrange
    from functools import partial
    from omegaconf import OmegaConf
    from pathlib import Path
    from tqdm import trange

    from lpdm.data import field_preprocess, get_label, get_well_dataset
    from lpdm.nn.autoencoder import get_autoencoder

    device = torch.device("cuda")

    # Config
    runpath = Path(run)
    runpath = runpath.expanduser().resolve(strict=True)

    cfg = OmegaConf.load(runpath / "config.yaml")

    cache_path = runpath / "cache" / physics / split / file

    print(f"caching {physics}/{split}/{file}")

    # Data
    dataset = get_well_dataset(
        path=datasets,
        physics=physics,
        split=split,
        steps=-1,
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

    state = torch.load(runpath / "state.pth", weights_only=True, map_location=device)

    autoencoder.load_state_dict(state)
    autoencoder.to(device)
    autoencoder.eval()

    del state

    # Encode
    if cache_path.exists():
        return
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(cache_path, mode="x") as f:
        for i in trange(len(dataset), ncols=88, ascii=True):
            visited = []

            for j in range(repeat):
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

                label = get_label(batch)

                if "state" not in f:
                    f.create_dataset(
                        "state",
                        shape=(len(dataset) * repeat, *z.shape),
                        dtype=np.float32,
                    )

                if "label" not in f:
                    f.create_dataset(
                        "label",
                        shape=(len(dataset) * repeat, *label.shape),
                        dtype=np.float32,
                    )

                f["state"][i * repeat + j] = z.numpy(force=True)
                f["label"][i * repeat + j] = label.numpy(force=True)


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", type=str)

    args = parser.parse_args()

    # Config
    cfg = compose(
        config_file="./configs/cache_latents.yaml",
        overrides=args.overrides,
    )

    # Files
    array = []

    for physic in cfg.dataset.physics:
        for split in [cfg.split]:
            path = os.path.join(cfg.server.datasets, physic, "data", split)
            files = glob.glob("*.hdf5", root_dir=path) + glob.glob("*.h5", root_dir=path)

            for file in files:
                array.append((physic, split, file))

    # Job(s)
    def launch(i: int):
        cache_latent(
            run=cfg.run,
            physics=array[i][0],
            split=array[i][1],
            file=array[i][2],
            augment=cfg.dataset.augment,
            repeat=cfg.repeat,
            datasets=cfg.server.datasets,
        )

    dawgz.schedule(
        dawgz.job(
            f=launch,
            name="cache_latent",
            array=len(array),
            cpus=cfg.compute.cpus,
            gpus=cfg.compute.gpus,
            ram=cfg.compute.ram,
            time=cfg.compute.time,
            partition=cfg.server.partition,
            constraint=cfg.server.constraint,
        ),
        name=f"cache {Path(cfg.run).name}",
        backend="slurm",
        env=[
            "export XDG_CACHE_HOME=$HOME/.cache",
        ],
    )
