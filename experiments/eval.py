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
    import math
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn

    from azula.sample import DDIMSampler, DDPMSampler, LMSSampler
    from einops import rearrange
    from functools import partial
    from omegaconf import OmegaConf
    from pathlib import Path

    from lpdm.data import field_preprocess, get_label, get_well_multi_dataset
    from lpdm.diffusion import MaskedDenoiser, get_denoiser
    from lpdm.nn.autoencoder import get_autoencoder
    from lpdm.plot import animate_fields

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
    rollout_steps = rollout // cfg.trajectory.stride + 1

    dataset = get_well_multi_dataset(
        path="/mnt/ceph/users/polymathic/the_well/datasets",
        physics=cfg.dataset.physics,
        split=split,
        steps=rollout_steps,
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

        if "predictor" in state:
            autoencoder = get_autoencoder(
                pix_channels=dataset.metadata.n_fields,
                out_channels=cfg.ae.predictor.cond_channels,
                **cfg.ae.ae,
            )

            autoencoder.load_state_dict(state["autoencoder"])
            autoencoder.cuda()
            autoencoder.eval()

            predictor = get_denoiser(
                shape=(dataset.metadata.n_fields, x.shape[-2], x.shape[-1]),
                **cfg.ae.predictor,
            ).to(device)

            predictor.load_state_dict(state["predictor"])
            predictor.cuda()
            predictor.eval()
        else:
            autoencoder = get_autoencoder(
                pix_channels=dataset.metadata.n_fields,
                **cfg.ae.ae,
            )

            autoencoder.load_state_dict(state)
            autoencoder.cuda()
            autoencoder.eval()

            predictor = None

        del state
    else:
        autoencoder = nn.Module()
        autoencoder.encode = nn.Identity()
        autoencoder.decode = nn.Identity()

        predictor = None

    ## Encode
    with torch.no_grad():
        temp = rearrange(x, "C L H W -> L C H W")
        temp = autoencoder.encode(temp)
        z = rearrange(temp, "L C H W -> C L H W")

    # Denoiser
    shape = (z.shape[0], cfg.trajectory.length, *z.shape[2:])

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

    # Inference
    def infer(mask, y):
        z = torch.zeros(shape, dtype=y.dtype, device=y.device)
        z[mask] = y.flatten()

        cond_denoiser = MaskedDenoiser(
            denoiser,
            y=z.flatten(),
            mask=mask.flatten(),
        )

        if sampling_alg == "ddpm":
            cond_sampler = DDPMSampler(cond_denoiser, steps=sampling_steps).to(device)
        elif sampling_alg == "ddim":
            cond_sampler = DDIMSampler(cond_denoiser, steps=sampling_steps).to(device)
        elif sampling_alg == "lms":
            cond_sampler = LMSSampler(cond_denoiser, steps=sampling_steps).to(device)

        z1 = cond_sampler.init((1, math.prod(shape)))
        z0 = cond_sampler(z1, label=label, cond=mask.reshape(1, *shape))
        z0 = z0.reshape(shape)

        with torch.no_grad():
            temp = rearrange(z0, "C L H W -> L C H W")
            temp = autoencoder.decode(temp)

            if predictor is None:
                x_hat = rearrange(temp, "L C H W -> C L H W")
            else:
                sampler = LMSSampler(predictor, steps=64).to(device)

                x1 = sampler.init((temp.shape[0], x.shape[0] * x.shape[2] * x.shape[3]))
                x0 = sampler(x1, cond=temp)
                x0 = x0.unflatten(-1, (x.shape[0], x.shape[2], x.shape[3]))

                x_hat = rearrange(x0, "L C H W -> C L H W")

        return x_hat, z0

    ## RNG
    if seed is None:
        seed = torch.initial_seed()

    _ = torch.manual_seed(seed + index)

    ## Observation
    mask = torch.zeros(shape, dtype=bool, device=device)
    mask[:, :context] = True

    y = z[:, :context]

    ## Rollout
    trajectory = []

    while len(trajectory) < rollout_steps:
        x_hat, z_hat = infer(mask, y)

        if trajectory:
            trajectory.extend(x_hat[:, overlap:].unbind(dim=1))
        else:
            trajectory.extend(x_hat.unbind(dim=1))

        y = z_hat[:, -overlap:]

        mask = torch.zeros(shape, dtype=bool, device=device)
        mask[:, :overlap] = True

    trajectory = trajectory[:rollout_steps]

    x_hat = torch.stack(trajectory, dim=1)

    # Evaluation
    lines = []

    for field in range(dataset.metadata.n_fields):
        for i in range(rollout_steps):
            u, v = x[field, i], x_hat[field, i]

            mse = torch.mean((u - v) ** 2)
            rmse = torch.sqrt(mse)
            nrmse = torch.sqrt(mse / torch.mean(u**2))
            vrmse = torch.sqrt(mse / torch.var(u))

            line = f"{runname},{split},{index},{seed},{(context - 1) * cfg.trajectory.stride + 1},{overlap},{sampling_alg},{sampling_steps},{field},{i * cfg.trajectory.stride},"
            line += f"{rmse},{nrmse},{vrmse},"
            line += ",".join(map(str, labels))
            line += "\n"

            lines.append(line)

    with open(outpath / "stats.csv", mode="a") as f:
        f.writelines(lines)

    # Viz
    plt.rcParams["animation.ffmpeg_path"] = "/mnt/sw/nix/store/fz8y69w4c97lcgv1wwk03bd4yh4zank7-ffmpeg-full-6.0-bin/bin/ffmpeg"  # fmt: off

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
            name="evaluate",
            array=len(array),
            cpus=cfg.compute.cpus,
            gpus=cfg.compute.gpus,
            ram=cfg.compute.ram,
            time=cfg.compute.time,
            partition=cfg.server.partition,
            constraint=cfg.server.constraint,
        ),
        name=f"evaluate {Path(cfg.run).name}",
        backend="slurm",
        env=[
            "export XDG_CACHE_HOME=$HOME/.cache",
        ],
    )
