#!/usr/bin/env python

import argparse
import os

from dawgz import job, schedule
from lpdm.hydra import multi_compose
from omegaconf import DictConfig


def train(
    cfg: DictConfig,
    autoencoder: str,
    datasets: str = "/mnt/ceph/users/polymathic/the_well/datasets",
):
    import torch
    import torch.distributed as dist
    import wandb

    from einops import rearrange
    from functools import partial
    from itertools import islice
    from lpdm.data import field_preprocess, get_well_dataset
    from lpdm.diffusion import DenoiserLoss, get_denoiser
    from lpdm.nn.autoencoder import AutoEncoder
    from lpdm.optim import ExponentialMovingAverage, get_optimizer, safe_gd_step
    from omegaconf import OmegaConf, open_dict
    from pathlib import Path
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler
    from tqdm import trange

    # DDP
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device_id = os.environ.get("LOCAL_RANK", rank)
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)

    # Config
    runid = wandb.util.generate_id()
    runname = f"{cfg.dataset.name}_{cfg.denoiser.name}_{cfg.optim.name}"

    runpath = Path(f"~/ceph/mpp-ldm/runs/{runname}_{runid}")
    runpath = runpath.expanduser().resolve()
    runpath.mkdir(parents=True, exist_ok=True)

    aepath = Path(autoencoder)
    aepath = aepath.expanduser().resolve(strict=True)

    with open_dict(cfg):
        cfg.path = str(runpath)
        cfg.ae = OmegaConf.load(aepath / "config.yaml").ae
        cfg.ae.path = str(aepath)

    if rank == 0:
        OmegaConf.save(cfg, runpath / "config.yaml")

    # Data
    trainset = get_well_dataset(
        path=os.path.join(datasets, cfg.dataset.physics, "data/train"),
        in_steps=cfg.trajectory.length,
        dt_stride=cfg.trajectory.stride,
        include_filters=cfg.dataset.include_filters,
    )

    train_sampler = DistributedSampler(
        dataset=trainset,
        rank=rank,
        drop_last=True,
        shuffle=True,
        seed=42,
    )

    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=cfg.train.batch_size,
        drop_last=True,
        shuffle=False,
        num_workers=32 // world_size,
        persistent_workers=True,
        pin_memory=True,
    )

    validset = get_well_dataset(
        path=os.path.join(datasets, cfg.dataset.physics, "data/valid"),
        in_steps=cfg.trajectory.length,
        include_filters=cfg.dataset.include_filters,
    )

    valid_sampler = DistributedSampler(
        dataset=validset,
        rank=rank,
        drop_last=True,
        shuffle=True,
        seed=42,
    )

    valid_loader = DataLoader(
        dataset=validset,
        sampler=valid_sampler,
        batch_size=cfg.train.batch_size,
        drop_last=True,
        shuffle=False,
        num_workers=32 // world_size,
        persistent_workers=True,
        pin_memory=True,
    )

    preprocess = partial(
        field_preprocess,
        mean=torch.as_tensor(cfg.dataset.stats.mean, device=device),
        std=torch.as_tensor(cfg.dataset.stats.std, device=device),
        transform=cfg.dataset.transform,
    )

    # Autoencoder
    autoencoder = AutoEncoder(
        pix_channels=trainset.metadata.n_fields,
        lat_channels=cfg.ae.lat_channels,
        hid_channels=cfg.ae.hid_channels,
        hid_blocks=cfg.ae.hid_blocks,
        attention_heads=cfg.ae.attention_heads,
        spectral_modes=cfg.ae.spectral_modes,
        spatial=2,
    )

    autoencoder.load_state_dict(torch.load(aepath / "state.pth"))
    autoencoder.to(device)
    autoencoder.eval()

    # Model, optimizer & scheduler
    denoiser = get_denoiser(
        shape=(cfg.ae.lat_channels, cfg.trajectory.length, 64, 64),
        label_features=trainset.metadata.n_constant_scalars,
        **cfg.denoiser,
    )

    model_loss = DenoiserLoss(denoiser=denoiser, a=3.0, b=3.0)
    model_loss = DistributedDataParallel(
        module=model_loss.to(device),
        device_ids=[device],
    )

    optimizer, scheduler = get_optimizer(
        params=model_loss.parameters(),
        optimizer=cfg.optim.optimizer,
        learning_rate=cfg.optim.learning_rate,
        weight_decay=cfg.optim.weight_decay,
        scheduler=cfg.optim.scheduler,
        epochs=cfg.train.epochs,
    )

    average = ExponentialMovingAverage(
        module=model_loss.module.denoiser,
        decay=cfg.optim.ema_decay,
    )

    # W&B
    if rank == 0:
        run = wandb.init(
            project="mpp-ldm-denoiser",
            id=runid,
            name=runname,
            config=OmegaConf.to_container(cfg),
        )

    # Training loop
    steps = cfg.train.epoch_size // cfg.train.batch_size // world_size

    if rank == 0:
        epochs = trange(cfg.train.epochs, ncols=88, miniters=1)
    else:
        epochs = range(cfg.train.epochs)

    for epoch in epochs:
        train_sampler.set_epoch(epoch)
        valid_sampler.set_epoch(epoch)

        ## Train
        model_loss.train()

        losses, grads = [], []

        for batch in islice(train_loader, steps):
            with torch.no_grad():
                x = batch["input_fields"]
                x = x.to(device, non_blocking=True)
                x = preprocess(x)
                x = rearrange(x, "B L H W C -> B L C H W")
                z = torch.func.vmap(autoencoder.encode)(x)
                z = rearrange(z, "B L C H W -> B C L H W")

            label = batch["constant_scalars"]
            label = label.to(device, non_blocking=True)

            loss = model_loss(z, label=label)
            loss.backward()

            grad_norm = safe_gd_step(optimizer, grad_clip=cfg.optim.grad_clip)

            average.update_parameters(model_loss.module.denoiser)

            losses.append(loss.detach())
            grads.append(grad_norm)

        losses = torch.stack(losses)
        grads = torch.stack(grads)

        if rank == 0:
            losses_list = [torch.empty_like(losses) for _ in range(world_size)]
            grads_list = [torch.empty_like(grads) for _ in range(world_size)]
        else:
            losses_list = None
            grads_list = None

        dist.gather(losses, losses_list, dst=0)
        dist.gather(grads, grads_list, dst=0)

        if rank == 0:
            losses = torch.cat(losses_list).cpu()
            grads = torch.cat(grads_list).cpu()

            logs = {}
            logs["train/loss/mean"] = losses.mean()
            logs["train/loss/std"] = losses.std()
            logs["train/grad_norm/mean"] = grads.mean()
            logs["train/grad_norm/std"] = grads.std()

        del losses, losses_list, grads, grads_list

        ## Eval
        model_loss.eval()

        losses = []

        with torch.no_grad():
            for batch in islice(valid_loader, steps):
                x = batch["input_fields"]
                x = x.to(device, non_blocking=True)
                x = preprocess(x)
                x = rearrange(x, "B L H W C -> B L C H W")
                z = torch.func.vmap(autoencoder.encode)(x)
                z = rearrange(z, "B L C H W -> B C L H W")

                label = batch["constant_scalars"]
                label = label.to(device, non_blocking=True)

                loss = model_loss(z, label=label)
                losses.append(loss)

        losses = torch.stack(losses)

        if rank == 0:
            losses_list = [torch.empty_like(losses) for _ in range(world_size)]
        else:
            losses_list = None

        dist.gather(losses, losses_list, dst=0)

        if rank == 0:
            losses = torch.stack(losses_list).cpu()

            logs["valid/loss/mean"] = losses.mean()
            logs["valid/loss/std"] = losses.std()

            run.log(logs)

        del losses, losses_list

        ## LR scheduler
        scheduler.step()

        ## Checkpoint
        if rank == 0:
            state = model_loss.module.denoiser.state_dict()
            state_ema = average.module.state_dict()

            torch.save(state, runpath / "state.pth")
            torch.save(state_ema, runpath / "state_ema.pth")

        dist.barrier()

    # W&B
    run.finish()

    # DDP
    dist.destroy_process_group()


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("autoencoder", type=str)
    parser.add_argument("overrides", nargs="*", type=str)
    parser.add_argument("--gpuxl", action="store_true", default=False)
    parser.add_argument("--slurm", action="store_true", default=False)

    args = parser.parse_args()

    # Config(s)
    configs = multi_compose(
        config_file="./configs/default_denoiser.yaml",
        overrides=args.overrides,
    )

    if args.gpuxl:
        datasets = "/mnt/gpuxl/polymathic/the_well/datasets"
    else:
        datasets = "/mnt/ceph/users/polymathic/the_well/datasets"

    def launch(i: int):
        train(configs[i], autoencoder=args.autoencoder, datasets=datasets)

    # Run
    if args.slurm:
        schedule(
            job(
                f=launch,
                name="train",
                array=len(configs),
                cpus=32,
                gpus=4,
                ram="256GB",
                time="2-00:00:00",
                partition="gpuxl" if args.gpuxl else "gpu",
                constraint="h100",
            ),
            name="training denoisers",
            backend="slurm",
            interpreter="torchrun --nnodes 1 --nproc-per-node 4 --standalone",
            env=[
                "export WANDB_SILENT=true",
                "export XDG_CACHE_HOME=$HOME/.cache",
            ],
        )
    else:
        for i in range(len(configs)):
            if i > 0:
                print("-" * 88)
            launch(i)
