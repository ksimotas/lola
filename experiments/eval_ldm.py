#!/usr/bin/env python

import argparse
import dawgz
import os

from lpdm.hydra import compose


def evaluate(
    run: str,
    target: str = "state_ema",
    split: str = "valid",
    context: int = 1,
    rollout: int = 64,
    sampling_alg: str = "lms",
    sampling_steps: int = 64,
    seed: int = 0,
):
    import math
    import matplotlib.pyplot as plt
    import torch

    from azula.sample import DDIMSampler, DDPMSampler, LMSSampler
    from einops import rearrange, reduce
    from functools import partial
    from omegaconf import OmegaConf
    from pathlib import Path

    from lpdm.data import field_preprocess, get_label, get_well_multi_dataset
    from lpdm.diffusion import MaskedDenoiser, get_autoencoder, get_denoiser
    from lpdm.plot import animate_fields

    device = torch.device("cuda")

    # RNG
    _ = torch.manual_seed(seed)

    # Config
    runpath = Path(run)
    runpath = runpath.expanduser().resolve()

    outpath = runpath / "results"
    outpath.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(runpath / "config.yaml")

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

    state = torch.load(runpath / "autoencoder/state.pth", weights_only=True)

    if "predictor" in state:
        autoencoder.load_state_dict(state["autoencoder"])
    else:
        autoencoder.load_state_dict(state)

    autoencoder.cuda()
    autoencoder.eval()

    # Item
    i = torch.randint(len(dataset), size=()).item()

    print(f"seed {seed} and item {split}/{i}")

    item = dataset[i]

    x = item["input_fields"]
    x = x.to(device)
    x = preprocess(x)

    label = get_label(item).to(device)

    ## Encode
    with torch.no_grad():
        temp = rearrange(x, "L H W C -> L C H W")
        temp = autoencoder.encode(temp)
        z = rearrange(temp, "L C H W -> C L H W")

    shape = z.shape
    label_features = label.numel()

    # Denoiser
    denoiser = get_denoiser(
        shape=shape,
        label_features=label_features,
        **cfg.denoiser,
    )

    denoiser.load_state_dict(torch.load(runpath / f"{target}.pth", weights_only=True))
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
        z0 = cond_sampler(z1, label=label)
        z0 = z0.reshape(shape)

        with torch.no_grad():
            temp = rearrange(z0, "C L H W -> L C H W")
            temp = autoencoder.decode(temp)
            x_hat = rearrange(temp, "L C H W -> L H W C")

        return x_hat, z0

    ## Observation
    mask = torch.zeros(z.shape, dtype=bool, device=device)
    mask[:, :context] = True

    y = z[:, :context]

    ## Rollout
    trajectory = []

    while len(trajectory) < rollout_steps:
        x_hat, z_hat = infer(mask, y)

        if trajectory:
            trajectory.extend(x_hat[context:].unbind(0))
        else:
            trajectory.extend(x_hat.unbind(0))

        y = z_hat[:, -context:]

    trajectory = trajectory[:rollout_steps]

    x_hat = torch.stack(trajectory)

    # Evaluation
    se = torch.square(x_hat - x)
    mse = reduce(se, "L H W C -> L", reduction="mean")
    rmse = torch.sqrt(mse)
    rmse = rmse.cpu().tolist()

    with open(outpath / "stats.csv", mode="a") as f:
        for i, e in enumerate(rmse):
            time = i * cfg.trajectory.stride
            f.write(f"{split},{seed},{context},{sampling_alg},{sampling_steps},{time},{e}\n")

    # Viz
    plt.rcParams["animation.ffmpeg_path"] = "/mnt/sw/nix/store/fz8y69w4c97lcgv1wwk03bd4yh4zank7-ffmpeg-full-6.0-bin/bin/ffmpeg"  # fmt: off

    figsize = (x.shape[1] / 128, x.shape[2] / 128)

    animation = animate_fields(x.cpu(), x_hat.cpu(), fields=cfg.dataset.fields, figsize=figsize)
    animation.save(outpath / f"{split}_{seed:03d}_{context}_{sampling_alg}_{sampling_steps}.mp4")


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", type=str)

    args = parser.parse_args()

    # Config
    cfg = compose(
        config_file="./configs/default_eval.yaml",
        overrides=args.overrides,
    )

    # Job
    def launch(i: int):
        evaluate(
            run=cfg.run,
            target=cfg.target,
            split=cfg.split,
            context=cfg.context,
            rollout=cfg.rollout,
            sampling_alg=cfg.sampling.alg,
            sampling_steps=cfg.sampling.steps,
            seed=i,
        )

    dawgz.schedule(
        dawgz.job(
            f=launch,
            name="evaluate",
            array=cfg.samples,
            cpus=4,
            gpus=1,
            ram="16GB",
            time="01:00:00",
            partition=cfg.server.partition,
            constraint=cfg.server.constraint,
        ),
        name=f"evaluate {os.path.basename(cfg.run)}",
        backend="slurm",
        env=[
            "export XDG_CACHE_HOME=$HOME/.cache",
        ],
    )
