#!/usr/bin/env python

import argparse
import dawgz
import random

from omegaconf import DictConfig
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

from lola.hydra import compose


def evaluate(
    run: str,
    server: DictConfig,
    indices: Sequence[Union[int, float]],
    target: str = "state",
    split: str = "valid",
    start: int = 0,
    context: int = 1,
    overlap: int = 1,
    samples: int = 1,
    filtering: Optional[str] = None,
    sampling: Dict[str, Any] = {},  # noqa: B006
    mixed_precision: bool = False,
    seed: Optional[int] = None,
    record: int = 0,
    **ignore,
):
    import time
    import torch
    import torch.nn as nn

    from azula.guidance import MMPSDenoiser
    from einops import rearrange, reduce
    from filelock import FileLock
    from functools import partial
    from omegaconf import OmegaConf
    from pathlib import Path
    from tqdm import tqdm

    from lola.autoencoder import get_autoencoder
    from lola.data import (
        field_postprocess,
        field_preprocess,
        get_well_inputs,
        get_well_multi_dataset,
    )
    from lola.diffusion import get_denoiser
    from lola.emulation import (
        decode_traj,
        emulate_diffusion,
        emulate_rollout,
        emulate_surrogate,
        encode_traj,
    )
    from lola.fourier import isotropic_cross_correlation, isotropic_power_spectrum
    from lola.plot import draw_movie
    from lola.surrogate import get_surrogate
    from lola.utils import randseed

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

    postprocess = partial(
        field_postprocess,
        mean=torch.as_tensor(cfg.dataset.stats.mean, device=device),
        std=torch.as_tensor(cfg.dataset.stats.std, device=device),
        transform=cfg.dataset.transform,
    )

    # Autoencoder
    if (runpath / "autoencoder").exists():
        cfg.ae = OmegaConf.load(runpath / "autoencoder/config.yaml").ae
        cfg.ae.latent_noise = 0.0

        state = torch.load(
            runpath / "autoencoder/state.pth", weights_only=True, map_location=device
        )

        autoencoder = get_autoencoder(
            pix_channels=dataset.metadata.n_fields,
            **cfg.ae,
        )

        autoencoder.load_state_dict(state)
        autoencoder.to(device)
        autoencoder.requires_grad_(False)
        autoencoder.eval()

        del state
    else:
        autoencoder = nn.Module()
        autoencoder.encode = nn.Identity()
        autoencoder.decode = nn.Identity()

    ## Get the latent shape and compression ratio
    x, label = get_well_inputs(dataset[0], device=device)

    # Emulator
    if hasattr(cfg, "denoiser"):
        denoiser = get_denoiser(
            channels=cfg.ae.lat_channels,
            label_features=label.numel(),
            spatial=3,
            masked=True,
            **cfg.denoiser,
        )

        denoiser.load_state_dict(
            torch.load(runpath / f"{target}.pth", weights_only=True, map_location=device),
        )
        denoiser.to(device)
        denoiser.requires_grad_(False)
        denoiser.eval()
    elif hasattr(cfg, "surrogate"):
        surrogate = get_surrogate(
            channels=cfg.ae.lat_channels,
            label_features=label.numel(),
            spatial=3,
            **cfg.surrogate,
        )

        surrogate.load_state_dict(
            torch.load(runpath / f"{target}.pth", weights_only=True, map_location=device)
        )
        surrogate.to(device)
        surrogate.eval()

    # RNG
    if seed is None:
        seed = torch.initial_seed()

    # Evaluation
    indices = {
        int(index * len(dataset)) if isinstance(index, float) else index for index in indices
    }

    for index in tqdm(indices, ncols=88, ascii=True):
        _ = torch.manual_seed(randseed(f"{seed},{index},{start}"))

        x, label = get_well_inputs(dataset[index], device=device)
        x = x[start :: cfg.trajectory.stride]
        x = preprocess(x)
        x = rearrange(x, "L H W C -> C L H W")

        with torch.no_grad():
            z = encode_traj(autoencoder, x)
            x_ae = decode_traj(autoencoder, z)

        compression = x.numel() / z.numel()

        ## Emulate
        if hasattr(cfg, "denoiser"):
            method = "diffusion"
            settings = f"{sampling.algorithm}{sampling.steps}"

            if filtering is None:
                emulate = lambda mask, z_obs, i: emulate_diffusion(
                    denoiser,
                    mask,
                    z_obs,
                    label=label,  # noqa: B023
                    **sampling,
                )
            else:
                # fmt: off
                def D(z):
                    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                        return decode_traj(autoencoder, z, batched=True)
                # fmt: on

                if filtering == "subsample":
                    A = lambda x: x[..., ::32, ::32]
                elif filtering == "downsample":
                    A = lambda x: reduce(x, "... (H h) (W w) -> ... H W", "mean", h=32, w=32)
                else:
                    raise ValueError(f"unknown operator '{filtering}'")

                y = A(x)
                y = y + 1e-1 * torch.randn_like(y)
                var_y = torch.tensor(1e-2, device=device)

                def emulate(mask, z_obs, i):
                    j = overlap if i > 0 else context
                    y_i = y[..., i + j : i + cfg.trajectory.length, :, :]  # noqa: B023
                    A_i = lambda z: A(D(z[..., j : j + len(y_i), :, :]))  # noqa: B023

                    return emulate_diffusion(
                        MMPSDenoiser(
                            denoiser,
                            y=y_i,
                            A=A_i,
                            var_y=var_y,  # noqa: B023
                            iterations=1,
                        ),
                        mask,
                        z_obs,
                        label=label,  # noqa: B023
                        **sampling,
                    )

        elif hasattr(cfg, "surrogate"):
            method = "surrogate"
            settings = None
            emulate = lambda mask, z_obs, i: emulate_surrogate(
                surrogate,
                mask,
                z_obs,
                label=label,  # noqa: B023
            )

        tic = time.time()

        with torch.autocast(device_type="cuda", enabled=mixed_precision):
            z_hat = emulate_rollout(
                emulate,
                z,
                window=cfg.trajectory.length,
                rollout=z.shape[1],
                context=context,
                overlap=overlap,
                batch=samples,
            )

        with torch.no_grad():
            x_hat = decode_traj(autoencoder, z_hat, batched=True)

        tac = time.time()

        del z_hat

        ## Speed
        speed = (tac - tic) / (x_hat.shape[0] * x_hat.shape[1])

        ## Postprocess
        x = postprocess(x, dim=-4)
        x_ae = postprocess(x_ae, dim=-4)
        x_hat = postprocess(x_hat, dim=-4)

        ## Stats
        lines = []

        for field in range(x.shape[0]):
            for t in range(context - 1, x.shape[1]):
                for auto_encoded in (False, True):
                    if auto_encoded:
                        u, v = x_ae[field, t], x_hat[:, field, t]
                    else:
                        u, v = x[field, t], x_hat[:, field, t]

                    # Spread
                    if samples > 1:
                        # see https://doi.org/10.1175/JHM-D-14-0008.1
                        spread = torch.mean(torch.square(v - torch.mean(v, dim=0)))
                        spread = torch.sqrt((samples + 1) / (samples - 1) * spread)
                    else:
                        spread = 0.0

                    # Skill
                    se = torch.square(u - torch.mean(v, dim=0))
                    mse = torch.mean(se)
                    rmse = torch.sqrt(mse)
                    nrmse = torch.sqrt(mse / torch.mean(u**2))
                    vrmse = torch.sqrt(mse / torch.var(u))

                    # Invariants
                    invariants = []

                    total_u = u.sum()
                    total_v = v.sum()

                    invariants.append(1 - total_v / total_u)

                    # Fourier
                    p_u, k = isotropic_power_spectrum(u, spatial=2)
                    p_v, _ = isotropic_power_spectrum(v, spatial=2)
                    p_v = torch.mean(p_v, dim=0)
                    c_uv, _ = isotropic_cross_correlation(u, v, spatial=2)
                    c_uv = torch.mean(c_uv, dim=0)

                    se_p = torch.square(1 - (p_v + 1e-6) / (p_u + 1e-6))
                    se_c = torch.square(1 - (c_uv + 1e-6) / torch.sqrt(p_u * p_v + 1e-12))

                    rmse_f = []

                    bins = torch.logspace(k[0].log2(), -1.0, steps=4, base=2)

                    for i in range(4):
                        if i < 3:
                            mask = torch.logical_and(bins[i] <= k, k <= bins[i + 1])
                        else:
                            mask = bins[i] <= k

                        rmse_f.append(torch.sqrt(torch.mean(se_p[mask])))
                        rmse_f.append(torch.sqrt(torch.mean(se_c[mask])))

                    # Write
                    line = f"{runname},{target},{compression},{method},{settings},{filtering},{speed},"
                    line += f"{split},{index},{start},{seed},"
                    line += f"{context},{overlap},{auto_encoded},"
                    line += f"{field},{(t - context + 1) * cfg.trajectory.stride},"
                    line += f"{spread},{rmse},{nrmse},{vrmse},"
                    line += ",".join(map(format, (*invariants, *rmse_f, *label.tolist())))
                    line += "\n"

                    lines.append(line)

        outdir = Path(f"{server.storage}/results/{cfg.dataset.name}")
        outdir = outdir.expanduser().resolve()
        (outdir / runname).mkdir(parents=True, exist_ok=True)

        with FileLock(outdir / "stats.csv.lock"):
            with open(outdir / "stats.csv", mode="a") as f:
                f.writelines(lines)

        # Video
        if x.shape[-1] < x.shape[-2]:
            x, x_hat = x.mT, x_hat.mT

        if record > 0:
            frames = torch.stack((x, *x_hat[:record]))
            frames = rearrange(frames, "N C L H W -> L N C H W")

            draw_movie(
                frames,
                file=(
                    outdir
                    / runname
                    / f"{runname}_{target}_{split}_{index:06d}_{start:03d}_{context}_{overlap}_{settings}_{filtering}_{seed}.mp4"
                ),
                fps=4.0 / cfg.trajectory.stride,
                isolate={2},
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
        evaluate(indices=array[i :: cfg.compute.jobs], **cfg)

    dawgz.schedule(
        dawgz.job(
            f=launch,
            name="eval",
            array=cfg.compute.jobs,
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
