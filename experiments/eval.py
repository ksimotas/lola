#!/usr/bin/env python

import argparse
import dawgz
import random

from pathlib import Path
from typing import Optional, Union

from lpdm.hydra import compose


def evaluate(
    run: str,
    target: str = "state",
    split: str = "valid",
    index: Union[int, float] = 0,
    context: int = 1,
    overlap: int = 1,
    rollout: int = 64,
    sampling_alg: str = "lms",
    sampling_steps: int = 64,
    seed: Optional[int] = None,
):
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn

    from einops import rearrange
    from functools import partial
    from omegaconf import OmegaConf
    from pathlib import Path

    from lpdm.data import field_preprocess, get_label, get_well_multi_dataset
    from lpdm.diffusion import get_denoiser
    from lpdm.emulation import (
        decode_traj,
        emulate_diffusion,
        emulate_rollout,
        emulate_surrogate,
        encode_traj,
    )
    from lpdm.nn.autoencoder import get_autoencoder
    from lpdm.plot import animate_fields
    from lpdm.surrogate import get_surrogate

    device = torch.device("cuda")

    # Config
    runpath = Path(run)
    runpath = runpath.expanduser().resolve()

    runname = runpath.name

    outpath = runpath / "results"
    outpath.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(runpath / "config.yaml")

    if hasattr(cfg, "ae_from"):
        cfg.ae = OmegaConf.load(runpath / "autoencoder/config.yaml")

    # Data
    rollout_strided = rollout // cfg.trajectory.stride + 1

    dataset = get_well_multi_dataset(
        path="/mnt/ceph/users/polymathic/the_well/datasets",
        physics=cfg.dataset.physics,
        split=split,
        steps=rollout_strided,
        min_dt_stride=cfg.trajectory.stride,
        max_dt_stride=cfg.trajectory.stride,
        include_filters=cfg.dataset.include_filters,
        augment=[s for s in cfg.dataset.augment if "random" not in s],
    )

    preprocess = partial(
        field_preprocess,
        mean=torch.as_tensor(cfg.dataset.stats.mean, device=device),
        std=torch.as_tensor(cfg.dataset.stats.std, device=device),
        transform=cfg.dataset.transform,
    )

    ## Item
    if isinstance(index, float):
        index = int(index * len(dataset))

    item = dataset[index]

    x = item["input_fields"]
    x = x.to(device)
    x = preprocess(x)
    x = rearrange(x, "L H W C -> C L H W")

    label = get_label(item).to(device)
    labels = label.tolist()

    # Autoencoder
    if hasattr(cfg, "ae_from"):
        state = torch.load(
            runpath / "autoencoder/state.pth", weights_only=True, map_location=device
        )

        autoencoder = get_autoencoder(
            pix_channels=dataset.metadata.n_fields,
            **cfg.ae.ae,
        )

        autoencoder.load_state_dict(state)
        autoencoder.cuda()
        autoencoder.eval()

        del state
    else:
        autoencoder = nn.Module()
        autoencoder.encode = nn.Identity()
        autoencoder.decode = nn.Identity()

    ## Encode
    with torch.no_grad():
        z = encode_traj(autoencoder, x)
        x_ae = decode_traj(autoencoder, z)

    # Emulator
    shape = (z.shape[0], cfg.trajectory.length, *z.shape[2:])

    if hasattr(cfg, "denoiser"):
        denoiser = get_denoiser(
            shape=shape,
            label_features=label.numel(),
            masked=True,
            **cfg.denoiser,
        )

        denoiser.load_state_dict(
            torch.load(runpath / f"{target}.pth", weights_only=True, map_location=device)
        )
        denoiser.cuda()
        denoiser.eval()
    elif hasattr(cfg, "surrogate"):
        surrogate = get_surrogate(
            shape=shape,
            label_features=label.numel(),
            **cfg.surrogate,
        ).to(device)

        surrogate.load_state_dict(
            torch.load(runpath / f"{target}.pth", weights_only=True, map_location=device)
        )
        surrogate.cuda()
        surrogate.eval()

    ## RNG
    if seed is None:
        seed = torch.initial_seed()

    _ = torch.manual_seed(seed + index)

    ## Rollout
    if hasattr(cfg, "denoiser"):
        emulate = lambda mask, z_obs: emulate_diffusion(
            denoiser,
            mask,
            z_obs,
            label=label,
            algorithm=sampling_alg,
            steps=sampling_steps,
        )
    elif hasattr(cfg, "surrogate"):
        emulate = lambda mask, z_obs: emulate_surrogate(
            surrogate,
            mask,
            z_obs,
            label=label,
        )

    z_hat = emulate_rollout(
        emulate,
        z,
        window=cfg.trajectory.length,
        rollout=rollout_strided,
        context=context,
        overlap=overlap,
    )

    with torch.no_grad():
        x_hat = decode_traj(autoencoder, z_hat)

    # Evaluation
    lines = []

    for field in range(dataset.metadata.n_fields):
        for i in range(rollout_strided):
            for auto_encoded in (False, True):
                if auto_encoded:
                    u, v = x_ae[field, i], x_hat[field, i]
                else:
                    u, v = x[field, i], x_hat[field, i]

                mse = torch.mean((u - v) ** 2)
                rmse = torch.sqrt(mse)
                nrmse = torch.sqrt(mse / torch.mean(u**2))
                vrmse = torch.sqrt(mse / torch.var(u))

                line = f"{runname},{split},{index},{seed},{(context - 1) * cfg.trajectory.stride + 1},{overlap},{sampling_alg},{sampling_steps},{field},{i * cfg.trajectory.stride},{auto_encoded},"
                line += f"{rmse},{nrmse},{vrmse},"
                line += ",".join(map(str, labels))
                line += "\n"

                lines.append(line)

    with open(outpath / "stats.csv", mode="a") as f:
        f.writelines(lines)

    # Viz
    plt.rcParams["animation.ffmpeg_path"] = "/mnt/sw/nix/store/fz8y69w4c97lcgv1wwk03bd4yh4zank7-ffmpeg-full-6.0-bin/bin/ffmpeg"  # fmt: off

    if x.shape[-1] < x.shape[-2]:
        x, x_hat = x.mT, x_hat.mT

    figsize = (x.shape[-1] / 64, x.shape[-2] / 64)

    animation = animate_fields(x.cpu(), x_hat.cpu(), fields=cfg.dataset.fields, figsize=figsize)
    animation.save(
        outpath
        / f"{runname}_{split}_{index:06d}_{seed}_{context}_{overlap}_{sampling_alg}_{sampling_steps}.mp4"
    )


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", type=str)

    args = parser.parse_args()

    # Config
    cfg = compose(
        config_file="./configs/eval.yaml",
        overrides=args.overrides,
    )

    ## RNG
    random.seed(cfg.seed)

    if cfg.array is None:
        array = [random.random() for _ in range(cfg.samples)]
    else:
        array = cfg.array

    # Job
    def launch(i: int):
        evaluate(
            run=cfg.run,
            target=cfg.target,
            split=cfg.split,
            index=array[i],
            context=cfg.context,
            overlap=cfg.overlap,
            rollout=cfg.rollout,
            sampling_alg=cfg.sampling.alg,
            sampling_steps=cfg.sampling.steps,
            seed=cfg.seed,
        )

    dawgz.schedule(
        dawgz.job(
            f=launch,
            name="eval",
            array=len(array),
            cpus=cfg.compute.cpus,
            gpus=cfg.compute.gpus,
            ram=cfg.compute.ram,
            time=cfg.compute.time,
            partition=cfg.server.partition,
            constraint=cfg.server.constraint,
        ),
        name=f"eval {Path(cfg.run).name}",
        backend="slurm",
        env=[
            "export XDG_CACHE_HOME=$HOME/.cache",
        ],
    )
