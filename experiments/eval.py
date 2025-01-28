#!/usr/bin/env python

import argparse
import dawgz
import random

from omegaconf import DictConfig
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

from lpdm.hydra import compose


def evaluate(
    run: str,
    server: DictConfig,
    target: str = "state",
    split: str = "valid",
    index: Union[int, float] = 0,
    start: int = 0,
    context: int = 1,
    overlap: int = 1,
    samples: int = 1,
    sampling: Dict[str, Any] = {},  # noqa: B006
    seed: Optional[int] = None,
    record: int = 1,
    # Ignore
    array: Sequence[Union[int, float]] = (),
    compute: Optional[DictConfig] = None,
):
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn

    from einops import rearrange
    from filelock import FileLock
    from functools import partial
    from omegaconf import OmegaConf
    from pathlib import Path

    from lpdm.data import field_preprocess, get_well_inputs, get_well_multi_dataset
    from lpdm.diffusion import get_denoiser
    from lpdm.emulation import (
        decode_traj,
        emulate_diffusion,
        emulate_rollout,
        emulate_surrogate,
        encode_traj,
    )
    from lpdm.fourier import isotropic_cross_correlation, isotropic_power_spectrum
    from lpdm.nn.autoencoder import get_autoencoder
    from lpdm.plot import animate_fields
    from lpdm.surrogate import get_surrogate

    device = torch.device("cuda")

    # Config
    runpath = Path(run)
    runpath = runpath.expanduser().resolve()

    runname = runpath.name

    cfg = OmegaConf.load(runpath / "config.yaml")

    # Data
    dataset = get_well_multi_dataset(
        path="/mnt/ceph/users/polymathic/the_well/datasets",
        physics=cfg.dataset.physics,
        split=split,
        steps=-1,
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

    x, label = get_well_inputs(dataset[index], device=device)
    x = x[start :: cfg.trajectory.stride]
    x = preprocess(x)
    x = rearrange(x, "L H W C -> C L H W")

    labels = label.tolist()

    # Autoencoder
    if (runpath / "autoencoder").exists():
        cfg.ae = OmegaConf.load(runpath / "autoencoder/config.yaml").ae
        state = torch.load(
            runpath / "autoencoder/state.pth", weights_only=True, map_location=device
        )

        autoencoder = get_autoencoder(
            pix_channels=dataset.metadata.n_fields,
            **cfg.ae,
        )

        autoencoder.load_state_dict(state)
        autoencoder.to(device)
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

    compression = x.numel() / z.numel()

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
        denoiser.to(device)
        denoiser.eval()
    elif hasattr(cfg, "surrogate"):
        surrogate = get_surrogate(
            shape=shape,
            label_features=label.numel(),
            **cfg.surrogate,
        )

        surrogate.load_state_dict(
            torch.load(runpath / f"{target}.pth", weights_only=True, map_location=device)
        )
        surrogate.to(device)
        surrogate.eval()

    ## RNG
    if seed is None:
        seed = torch.initial_seed()

    _ = torch.manual_seed(seed + 101 * index + start)

    ## Emulation
    if hasattr(cfg, "denoiser"):
        method = "diffusion"
        emulate = lambda mask, z_obs: emulate_diffusion(
            denoiser,
            mask,
            z_obs,
            label=label,
            **sampling,
        )
    elif hasattr(cfg, "surrogate"):
        method = "surrogate"
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
        rollout=z.shape[1],
        context=(context - 1) // cfg.trajectory.stride + 1,
        overlap=overlap,
        batch=samples,
    )

    with torch.no_grad():
        x_hat = decode_traj(autoencoder, z_hat, batched=True)

    del z_hat

    # Evaluation
    lines = []

    for field in range(x.shape[0]):
        for t in range(x.shape[1]):
            for auto_encoded in (False, True):
                if auto_encoded:
                    u, v = x_ae[field, t], x_hat[:, field, t]
                else:
                    u, v = x[field, t], x_hat[:, field, t]

                # Spread
                if samples > 1:
                    spread = torch.sqrt(torch.mean(torch.square(v - torch.mean(v, dim=0))))
                else:
                    spread = 0.0

                # Skill
                se = torch.square(u - torch.mean(v, dim=0))
                mse = torch.mean(se)
                rmse = torch.sqrt(mse)
                nrmse = torch.sqrt(mse / torch.mean(u**2))
                vrmse = torch.sqrt(mse / torch.var(u))

                # Fourier
                p_u, k = isotropic_power_spectrum(u, spatial=2)
                p_v, _ = isotropic_power_spectrum(v, spatial=2)
                p_v = torch.mean(p_v, dim=0)
                c_uv, _ = isotropic_cross_correlation(u, v, spatial=2)
                c_uv = torch.mean(c_uv, dim=0)

                sre_p = torch.square(1 - p_v / p_u)
                sre_c = torch.square(1 - c_uv / torch.sqrt(p_u * p_v))

                rmsre_f = []

                bins = torch.logspace(k[0].log2(), -1.0, steps=4, base=2)

                for i in range(4):
                    if i < 3:
                        mask = torch.logical_and(bins[i] <= k, k <= bins[i + 1])
                    else:
                        mask = bins[i] <= k

                    rmsre_f.append(torch.sqrt(torch.mean(sre_p[mask])))
                    rmsre_f.append(torch.sqrt(torch.mean(sre_c[mask])))

                # Write
                line = f"{runname},{target},{method},{compression},"
                line += f"{split},{index},{start},{seed},{context},{overlap},{auto_encoded},{field},{t * cfg.trajectory.stride},"
                line += f"{spread},{rmse},{nrmse},{vrmse},"
                line += ",".join(map(format, (*rmsre_f, *labels)))
                line += "\n"

                lines.append(line)

    outdir = Path(f"{server.storage}/results/{cfg.dataset.name}")
    outdir = outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    with FileLock(outdir / "stats.csv.lock"):
        with open(outdir / "stats.csv", mode="a") as f:
            f.writelines(lines)

    # Video
    plt.rcParams["animation.ffmpeg_path"] = "/mnt/sw/nix/store/fz8y69w4c97lcgv1wwk03bd4yh4zank7-ffmpeg-full-6.0-bin/bin/ffmpeg"  # fmt: off

    if x.shape[-1] < x.shape[-2]:
        x, x_hat = x.mT, x_hat.mT

    for i in range(min(samples, record)):
        animation = animate_fields(
            x,
            x_hat[i],
            fields=cfg.dataset.fields,
            figsize=(x.shape[-1] / 64, x.shape[-2] / 64),
        )
        animation.save(
            outdir
            / f"{runname}_{target}_{split}_{index:06d}_{start:03d}_{seed}_{context}_{overlap}_{i:03d}.mp4"
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

    if isinstance(cfg.array, int):
        array = [random.random() for _ in range(cfg.array)]
    else:
        array = cfg.array

    # Job
    def launch(i: int):
        evaluate(index=array[i], **cfg)

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
            exclude=cfg.server.exclude,
        ),
        name=f"eval {Path(cfg.run).name}",
        backend="slurm",
        env=[
            "export XDG_CACHE_HOME=$HOME/.cache",
        ],
    )
