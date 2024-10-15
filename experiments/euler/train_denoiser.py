#!/usr/bin/env python

import argparse
import dawgz
import wandb

from functools import partial
from omegaconf import DictConfig

from lpdm.hydra import compose


def train(runid: str, cfg: DictConfig):
    import os
    import torch
    import torch.distributed as dist
    import wandb

    from einops import rearrange
    from itertools import islice
    from omegaconf import OmegaConf, open_dict
    from pathlib import Path
    from torch.nn.parallel import DistributedDataParallel
    from tqdm import trange

    from lpdm.data import MiniWellDataset, find_hdf5, get_dataloader
    from lpdm.diffusion import DenoiserLoss, get_denoiser
    from lpdm.optim import ExponentialMovingAverage, get_optimizer, safe_gd_step
    from lpdm.utils import map_to_memory, process_cpu_count, randseed

    # DDP
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device_id = os.environ.get("LOCAL_RANK", rank)
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)

    # Config
    runname = f"{cfg.denoiser.name}_{cfg.optim.name}"

    runpath = Path(f"~/ceph/mpp-ldm/runs/{runid}_{runname}")
    runpath = runpath.expanduser().resolve()
    runpath.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        os.symlink(os.path.realpath(cfg.ae_path, strict=True), runpath / "autoencoder")

    dist.barrier()

    with open_dict(cfg):
        cfg.path = str(runpath)
        cfg.seed = randseed(runid)
        cfg.ae = OmegaConf.load(runpath / "autoencoder/config.yaml").ae

    if rank == 0:
        OmegaConf.save(cfg, runpath / "config.yaml")

    # Data
    train_files = find_hdf5(
        path=runpath / "autoencoder/cache" / cfg.dataset.physics / "train",
        include_filters=cfg.dataset.include_filters,
    )

    valid_files = find_hdf5(
        path=runpath / "autoencoder/cache" / cfg.dataset.physics / "valid",
        include_filters=cfg.dataset.include_filters,
    )

    if rank == 0:
        for file in (*train_files, *valid_files):
            map_to_memory(file, shm=f"/dev/shm/{runid}", exist_ok=False)

    dist.barrier()

    train_files = [
        map_to_memory(file, shm=f"/dev/shm/{runid}", exist_ok=True) for file in train_files
    ]

    trainset = MiniWellDataset.from_files(
        files=train_files,
        steps=cfg.trajectory.length,
        stride=cfg.trajectory.stride,
    )

    train_loader, train_sampler = get_dataloader(
        dataset=trainset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=process_cpu_count() // world_size,
        rank=rank,
        world_size=world_size,
        seed=cfg.seed,
    )

    valid_files = [
        map_to_memory(file, shm=f"/dev/shm/{runid}", exist_ok=True) for file in valid_files
    ]

    validset = MiniWellDataset.from_files(
        files=valid_files,
        steps=cfg.trajectory.length,
        stride=cfg.trajectory.stride,
    )

    valid_loader, valid_sampler = get_dataloader(
        dataset=validset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=process_cpu_count() // world_size,
        rank=rank,
        world_size=world_size,
        seed=cfg.seed,
    )

    item = validset[0]
    item["state"] = rearrange(item["state"], "L C H W -> C L H W")

    # Model, optimizer & scheduler
    denoiser = get_denoiser(
        shape=item["state"].shape,
        label_features=item["label"].numel(),
        **cfg.denoiser,
    )

    if cfg.boot_state:
        denoiser.load_state_dict(torch.load(cfg.boot_state))

    model_loss = DenoiserLoss(denoiser=denoiser, a=3.0, b=3.0)
    model_loss = DistributedDataParallel(
        module=model_loss.to(device),
        device_ids=[device],
    )

    optimizer, scheduler = get_optimizer(
        params=model_loss.parameters(),
        epochs=cfg.train.epochs,
        **cfg.optim,
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
        epochs = trange(cfg.train.epochs, ncols=88, ascii=True)
    else:
        epochs = range(cfg.train.epochs)

    best_valid_loss = float("inf")

    for epoch in epochs:
        train_sampler.set_epoch(epoch)
        valid_sampler.set_epoch(epoch)

        ## Train
        model_loss.train()

        losses, grads = [], []

        for batch in islice(train_loader, steps):
            z = batch["state"]
            z = z.to(device, non_blocking=True)
            z = rearrange(z, "B L C H W -> B C L H W")

            label = batch["label"]
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
            logs["train/loss/mean"] = losses.mean().item()
            logs["train/loss/std"] = losses.std().item()
            logs["train/grad_norm/mean"] = grads.mean().item()
            logs["train/grad_norm/std"] = grads.std().item()
            logs["train/learning_rate"] = optimizer.param_groups[0]["lr"]

        del losses, losses_list, grads, grads_list

        ## Eval
        model_loss.eval()

        losses = []

        with torch.no_grad():
            for batch in islice(valid_loader, steps):
                z = batch["state"]
                z = z.to(device, non_blocking=True)
                z = rearrange(z, "B L C H W -> B C L H W")

                label = batch["label"]
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

            logs["valid/loss/mean"] = losses.mean().item()
            logs["valid/loss/std"] = losses.std().item()

            epochs.set_postfix(
                lt=logs["train/loss/mean"],
                lv=logs["valid/loss/mean"],
            )

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

            if logs["valid/loss/mean"] < best_valid_loss:
                best_valid_loss = logs["valid/loss/mean"]

                torch.save(state, runpath / "state_best.pth")
                torch.save(state_ema, runpath / "state_best_ema.pth")

        dist.barrier()

    # W&B
    run.finish()

    # DDP
    dist.destroy_process_group()


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", type=str)
    parser.add_argument("--cpus-per-gpu", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--gpuxl", action="store_true", default=False)
    parser.add_argument("--ram", type=str, default="256GB")
    parser.add_argument("--time", type=str, default="7-00:00:00")

    args = parser.parse_args()

    # Config
    cfg = compose(
        config_file="./configs/default_denoiser.yaml",
        overrides=args.overrides,
    )

    # Job
    runid = wandb.util.generate_id()

    dawgz.schedule(
        dawgz.job(
            f=partial(train, runid, cfg),
            name="train",
            cpus=args.cpus_per_gpu * args.gpus,
            gpus=args.gpus,
            ram=args.ram,
            time=args.time,
            partition="gpuxl" if args.gpuxl else "gpu",
            constraint="h100",
            exclude="workergpu166",
        ),
        name=f"training denoiser {runid}",
        backend="slurm",
        interpreter=f"torchrun --nnodes 1 --nproc-per-node {args.gpus} --standalone",
        env=[
            "export WANDB_SILENT=true",
            "export XDG_CACHE_HOME=$HOME/.cache",
        ],
    )
