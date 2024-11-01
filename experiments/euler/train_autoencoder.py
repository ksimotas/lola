#!/usr/bin/env python

import argparse
import dawgz
import wandb

from functools import partial
from omegaconf import DictConfig

from lpdm.hydra import compose


def train(
    runid: str,
    cfg: DictConfig,
    datasets: str = "/mnt/ceph/users/polymathic/the_well/datasets",
):
    import os
    import torch
    import torch.distributed as dist
    import wandb

    from einops import rearrange
    from functools import partial
    from omegaconf import OmegaConf, open_dict
    from pathlib import Path
    from torch.nn.parallel import DistributedDataParallel
    from tqdm import trange

    from lpdm.data import field_preprocess, get_dataloader, get_well_dataset
    from lpdm.loss import WeightedLoss
    from lpdm.nn.autoencoder import AutoEncoder
    from lpdm.optim import get_optimizer, safe_gd_step
    from lpdm.utils import process_cpu_count, randseed

    # DDP
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device_id = os.environ.get("LOCAL_RANK", rank)
    device_id = int(device_id)
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)

    # Config
    assert cfg.train.batch_size % world_size == 0
    assert cfg.train.epoch_size % (cfg.train.batch_size * cfg.train.accumulation) == 0

    runname = f"{cfg.ae.name}_{cfg.optim.name}"

    runpath = Path(f"~/ceph/mpp-ldm/runs/{runid}_{runname}")
    runpath = runpath.expanduser().resolve()
    runpath.mkdir(parents=True, exist_ok=True)

    with open_dict(cfg):
        cfg.path = str(runpath)
        cfg.seed = randseed(runid)

    if rank == 0:
        OmegaConf.save(cfg, runpath / "config.yaml")

    # Data
    trainset = get_well_dataset(
        path=os.path.join(datasets, cfg.dataset.physics),
        split="train",
        steps=1,
        include_filters=cfg.dataset.include_filters,
        augment=cfg.dataset.augment,
    )

    train_loader = get_dataloader(
        dataset=trainset,
        batch_size=cfg.train.batch_size // world_size,
        shuffle="lazy" if cfg.train.lazy_shuffle else True,
        infinite=True,
        num_workers=process_cpu_count() // world_size,
        rank=rank,
        world_size=world_size,
        seed=cfg.seed,
    )

    validset = get_well_dataset(
        path=os.path.join(datasets, cfg.dataset.physics),
        split="valid",
        steps=1,
        include_filters=cfg.dataset.include_filters,
        augment=cfg.dataset.augment,
    )

    valid_loader = get_dataloader(
        dataset=validset,
        batch_size=cfg.train.batch_size // world_size,
        shuffle=False,
        infinite=True,
        num_workers=process_cpu_count() // world_size,
        rank=rank,
        world_size=world_size,
        seed=cfg.seed,
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
    ).to(device)

    autoencoder_loss = WeightedLoss(**cfg.ae.loss).to(device)

    if cfg.boot_state:
        autoencoder.load_state_dict(torch.load(cfg.boot_state, weights_only=True))

    autoencoder = DistributedDataParallel(
        module=autoencoder,
        device_ids=[device_id],
    )

    optimizer, scheduler = get_optimizer(
        params=autoencoder.parameters(),
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
    if rank == 0:
        epochs = trange(cfg.train.epochs, ncols=88, ascii=True)
    else:
        epochs = range(cfg.train.epochs)

    steps_per_epoch = cfg.train.epoch_size // cfg.train.batch_size // cfg.train.accumulation

    for epoch in epochs:
        ## Train
        autoencoder.train()

        losses, grads = [], []

        for i in range(cfg.train.epoch_size // cfg.train.batch_size):
            batch = next(train_loader)

            x = batch["input_fields"]
            x = x.to(device, non_blocking=True)
            x = preprocess(x)
            x = rearrange(x, "B 1 H W C -> B C H W")

            if (i + 1) % cfg.train.accumulation == 0:
                y, z = autoencoder(x)
                loss = autoencoder_loss(x, y)
                loss.backward()

                grad_norm = safe_gd_step(optimizer, grad_clip=cfg.optim.grad_clip)
                grads.append(grad_norm)
            else:
                with autoencoder.no_sync():
                    y, z = autoencoder(x)
                    loss = autoencoder_loss(x, y)
                    loss.backward()

            losses.append(loss.detach())

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
            logs["train/update_steps"] = (epoch + 1) * steps_per_epoch
            logs["train/samples"] = (epoch + 1) * cfg.train.epoch_size

        del losses, losses_list, grads, grads_list

        ## Eval
        autoencoder.eval()

        losses = []

        with torch.no_grad():
            for _ in range(cfg.train.epoch_size // cfg.train.batch_size):
                batch = next(valid_loader)

                x = batch["input_fields"]
                x = x.to(device, non_blocking=True)
                x = preprocess(x)
                x = rearrange(x, "B 1 H W C -> B C H W")

                y, z = autoencoder(x)
                loss = autoencoder_loss(x, y)
                losses.append(loss)

        losses = torch.stack(losses)

        if rank == 0:
            losses_list = [torch.empty_like(losses) for _ in range(world_size)]
        else:
            losses_list = None

        dist.gather(losses, losses_list, dst=0)

        if rank == 0:
            losses = torch.cat(losses_list).cpu()

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
            state = autoencoder.module.state_dict()
            torch.save(state, runpath / "state.pth")

        dist.barrier(device_ids=[device_id])

    # W&B
    if rank == 0:
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
        config_file="../configs/default_autoencoder.yaml",
        overrides=args.overrides,
    )

    if args.gpuxl:
        datasets = "/mnt/gpuxl/polymathic/the_well/datasets"
    else:
        datasets = "/mnt/ceph/users/polymathic/the_well/datasets"

    # Job
    runid = wandb.util.generate_id()

    dawgz.schedule(
        dawgz.job(
            f=partial(train, runid, cfg, datasets),
            name="train",
            cpus=args.cpus_per_gpu * args.gpus,
            gpus=args.gpus,
            ram=args.ram,
            time=args.time,
            partition="gpuxl" if args.gpuxl else "gpu",
            constraint="h100|a100-80gb",
            exclude="workergpu166",
        ),
        name=f"training auto-encoder {runid}",
        backend="slurm",
        interpreter=f"torchrun --nnodes 1 --nproc-per-node {args.gpus} --standalone",
        env=[
            "export WANDB_SILENT=true",
            "export XDG_CACHE_HOME=$HOME/.cache",
        ],
    )
