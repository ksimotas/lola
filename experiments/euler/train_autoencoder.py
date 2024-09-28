#!/usr/bin/env python

import argparse
import os

from dawgz import job, schedule
from lpdm.hydra import multi_compose
from omegaconf import DictConfig


def train(
    cfg: DictConfig,
    datasets: str = "/mnt/ceph/users/polymathic/the_well/datasets",
):
    import torch
    import torch.distributed as dist
    import wandb

    from einops import rearrange
    from functools import partial
    from itertools import islice
    from lpdm.data import field_preprocess, get_dataloader, get_well_dataset
    from lpdm.nn.autoencoder import AutoEncoder, AutoEncoderLoss
    from lpdm.optim import get_optimizer, safe_gd_step
    from lpdm.utils import process_cpu_count
    from omegaconf import OmegaConf, open_dict
    from pathlib import Path
    from torch.nn.parallel import DistributedDataParallel
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
    runname = f"{cfg.dataset.name}_{cfg.ae.name}_{cfg.optim.name}"

    runpath = Path(f"~/ceph/mpp-ldm/runs/{runname}_{runid}")
    runpath = runpath.expanduser().resolve()
    runpath.mkdir(parents=True, exist_ok=True)

    with open_dict(cfg):
        cfg.path = str(runpath)

    if rank == 0:
        OmegaConf.save(cfg, runpath / "config.yaml")

    # Data
    trainset = get_well_dataset(
        path=os.path.join(datasets, cfg.dataset.physics),
        split="train",
        steps=1,
        include_filters=cfg.dataset.include_filters,
    )

    train_loader, train_sampler = get_dataloader(
        dataset=trainset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=32 // world_size,
        rank=rank,
        world_size=world_size,
        seed=42,
    )

    validset = get_well_dataset(
        path=os.path.join(datasets, cfg.dataset.physics),
        split="valid",
        steps=1,
        include_filters=cfg.dataset.include_filters,
    )

    valid_loader, valid_sampler = get_dataloader(
        dataset=validset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=process_cpu_count() // world_size,
        rank=rank,
        world_size=world_size,
        seed=42,
    )

    preprocess = partial(
        field_preprocess,
        mean=torch.as_tensor(cfg.dataset.stats.mean, device=device),
        std=torch.as_tensor(cfg.dataset.stats.std, device=device),
        transform=cfg.dataset.transform,
    )

    # Model, optimizer & scheduler
    autoencoder = AutoEncoder(
        pix_channels=trainset.metadata.n_fields,
        **cfg.ae,
    )

    model_loss = AutoEncoderLoss(
        autoencoder=autoencoder,
        **cfg.ae.loss,
    )

    model_loss = DistributedDataParallel(
        module=model_loss.to(device),
        device_ids=[device],
    )

    optimizer, scheduler = get_optimizer(
        params=model_loss.parameters(),
        epochs=cfg.train.epochs,
        **cfg.optim,
    )

    # W&B
    if rank == 0:
        run = wandb.init(
            project="mpp-ldm-ae",
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
            x = batch["input_fields"]
            x = x.to(device, non_blocking=True)
            x = preprocess(x)
            x = rearrange(x, "B 1 H W C -> B C H W")

            loss, y = model_loss(x)
            loss.backward()

            grad_norm = safe_gd_step(optimizer, grad_clip=cfg.optim.grad_clip)

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
                x = rearrange(x, "B 1 H W C -> B C H W")

                loss, y = model_loss(x)
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
            state = model_loss.module.autoencoder.state_dict()
            torch.save(state, runpath / "state.pth")

        dist.barrier()

    # W&B
    run.finish()

    # DDP
    dist.destroy_process_group()


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", type=str)
    parser.add_argument("--cpus", type=int, default=32)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--gpuxl", action="store_true", default=False)
    parser.add_argument("--ram", type=str, default="256GB")
    parser.add_argument("--time", type=str, default="2-00:00:00")

    args = parser.parse_args()

    # Config(s)
    configs = multi_compose(
        config_file="./configs/default_autoencoder.yaml",
        overrides=args.overrides,
    )

    if args.gpuxl:
        datasets = "/mnt/gpuxl/polymathic/the_well/datasets"
    else:
        datasets = "/mnt/ceph/users/polymathic/the_well/datasets"

    def launch(i: int):
        train(configs[i], datasets=datasets)

    schedule(
        job(
            f=launch,
            name="train",
            array=len(configs),
            cpus=args.cpus,
            gpus=args.gpu,
            ram=args.ram,
            time=args.time,
            partition="gpuxl" if args.gpuxl else "gpu",
            constraint="h100",
        ),
        name="training auto-encoders",
        backend="slurm",
        interpreter=f"torchrun --nnodes 1 --nproc-per-node {args.gpus} --standalone",
        env=[
            "export WANDB_SILENT=true",
            "export XDG_CACHE_HOME=$HOME/.cache",
        ],
    )
