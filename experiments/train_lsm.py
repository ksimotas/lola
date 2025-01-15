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
    from omegaconf import OmegaConf, open_dict
    from pathlib import Path
    from torch.nn.parallel import DistributedDataParallel
    from tqdm import trange

    from lpdm.data import MiniWellDataset, find_hdf5, get_dataloader, random_context_mask
    from lpdm.optim import get_optimizer, safe_gd_step
    from lpdm.surrogate import get_surrogate
    from lpdm.utils import map_to_memory, randseed

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

    runname = f"{cfg.dataset.name}_{cfg.surrogate.name}_{cfg.optim.name}"

    runpath = Path(f"~/ceph/mpp-ldm/runs/lsm/{runid}_{runname}")
    runpath = runpath.expanduser().resolve()
    runpath.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        os.symlink(os.path.realpath(cfg.ae_from, strict=True), runpath / "autoencoder")

    dist.barrier(device_ids=[device_id])

    with open_dict(cfg):
        cfg.path = str(runpath)
        cfg.seed = randseed(runid)

    if rank == 0:
        OmegaConf.save(cfg, runpath / "config.yaml")

    # Stem
    if cfg.fork_from is None:
        counter = {
            "epoch": 0,
            "update_samples": 0,
            "update_steps": 0,
        }
    else:
        stem = wandb.Api().run(path=cfg.fork_from)
        stem_dir = Path(stem.config["path"]).name
        stem_path = Path(f"~/ceph/mpp-ldm/runs/lsm/{stem_dir}")
        stem_path = stem_path.expanduser().resolve()
        stem_state = torch.load(
            stem_path / f"{cfg.fork_target}.pth", weights_only=True, map_location=device
        )

        counter = {
            "epoch": stem.summary["_step"] + 1,
            "update_samples": stem.summary["train/samples"],
            "update_steps": stem.summary["train/update_steps"],
        }

    # Data
    files = {
        split: [
            file
            for physic in cfg.dataset.physics
            for file in find_hdf5(
                path=runpath / "autoencoder/cache" / physic / split,
                include_filters=cfg.dataset.include_filters,
            )
        ]
        for split in ("train", "valid")
    }

    if rank == 0:
        for split in ("train", "valid"):
            for file in files[split]:
                map_to_memory(file, shm=f"/dev/shm/{runid}", exist_ok=False)

    dist.barrier(device_ids=[device_id])

    files = {
        split: [
            map_to_memory(file, shm=f"/dev/shm/{runid}", exist_ok=True) for file in files[split]
        ]
        for split in ("train", "valid")
    }

    trainset = MiniWellDataset.from_files(
        files=files["train"],
        steps=cfg.trajectory.length,
        stride=cfg.trajectory.stride,
    )

    train_loader = get_dataloader(
        dataset=trainset,
        batch_size=cfg.train.batch_size // world_size,
        shuffle=True,
        infinite=True,
        num_workers=cfg.compute.cpus_per_gpu,
        rank=rank,
        world_size=world_size,
        seed=cfg.seed,
    )

    validset = MiniWellDataset.from_files(
        files=files["valid"],
        steps=cfg.trajectory.length,
        stride=cfg.trajectory.stride,
    )

    valid_loader = get_dataloader(
        dataset=validset,
        batch_size=cfg.train.batch_size // world_size,
        shuffle=True,
        infinite=True,
        num_workers=cfg.compute.cpus_per_gpu,
        rank=rank,
        world_size=world_size,
        seed=cfg.seed,
    )

    batch = next(valid_loader)
    batch["state"] = rearrange(batch["state"], "B L C H W -> B C L H W")

    # Model, optimizer & scheduler
    surrogate = get_surrogate(
        shape=batch["state"].shape[1:],
        label_features=batch["label"].shape[1],
        **cfg.surrogate,
    ).to(device)

    if cfg.fork_from is not None:
        surrogate.load_state_dict(stem_state)
        del stem_state

    surrogate = DistributedDataParallel(
        module=surrogate,
        device_ids=[device_id],
    )

    optimizer, scheduler = get_optimizer(
        params=surrogate.parameters(),
        epochs=cfg.train.epochs,
        **cfg.optim,
    )

    # W&B
    if rank == 0:
        run = wandb.init(
            entity=cfg.wandb.entity,
            project="mpp-lsm",
            id=runid,
            name=runname,
            config=OmegaConf.to_container(cfg),
        )

    # Training loop
    if rank == 0:
        epochs = trange(cfg.train.epochs, ncols=88, ascii=True)
    else:
        epochs = range(cfg.train.epochs)

    best_valid_loss = float("inf")

    for _ in epochs:
        ## Train
        surrogate.train()

        losses, grads = [], []

        for i in range(cfg.train.epoch_size // cfg.train.batch_size):
            batch = next(train_loader)

            z = batch["state"]
            z = z.to(device, non_blocking=True)
            z = rearrange(z, "B L C H W -> B C L H W")

            label = batch["label"]
            label = label.to(device, non_blocking=True)

            mask = random_context_mask(z, rho=0.66, atleast=1)

            if (i + 1) % cfg.train.accumulation == 0:
                y = surrogate(z * mask, mask=mask, label=label)

                loss = (z - y).square().mean()
                loss_acc = loss / cfg.train.accumulation
                loss_acc.backward()

                grad_norm = safe_gd_step(optimizer, grad_clip=cfg.optim.grad_clip)
                grads.append(grad_norm)

                counter["update_samples"] += cfg.train.batch_size * cfg.train.accumulation
                counter["update_steps"] += 1
            else:
                with surrogate.no_sync():
                    y = surrogate(z * mask, mask=mask, label=label)

                    loss = (z - y).square().mean()
                    loss_acc = loss / cfg.train.accumulation
                    loss_acc.backward()

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
            logs["train/update_steps"] = counter["update_steps"]
            logs["train/samples"] = counter["update_samples"]

        del losses, losses_list, grads, grads_list

        ## Eval
        surrogate.eval()

        losses = []

        with torch.no_grad():
            for _ in range(cfg.train.epoch_size // cfg.train.batch_size):
                batch = next(valid_loader)

                z = batch["state"]
                z = z.to(device, non_blocking=True)
                z = rearrange(z, "B L C H W -> B C L H W")

                label = batch["label"]
                label = label.to(device, non_blocking=True)

                mask = random_context_mask(z, rho=0.66, atleast=1)

                y = surrogate(z * mask, mask=mask, label=label)

                loss = (z - y).square().mean()
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

            run.log(logs, step=counter["epoch"])

            counter["epoch"] += 1

        del losses, losses_list

        ## LR scheduler
        scheduler.step()

        ## Checkpoint
        if rank == 0:
            state = surrogate.module.state_dict()

            torch.save(state, runpath / "state.pth")

            if logs["valid/loss/mean"] < best_valid_loss:
                best_valid_loss = logs["valid/loss/mean"]

                torch.save(state, runpath / "state_best.pth")

            del state

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

    args = parser.parse_args()

    # Config
    cfg = compose(
        config_file="./configs/train_lsm.yaml",
        overrides=args.overrides,
    )

    # Job
    runid = wandb.util.generate_id()

    dawgz.schedule(
        dawgz.job(
            f=partial(train, runid, cfg),
            name=f"lsm {runid}",
            cpus=cfg.compute.cpus_per_gpu * cfg.compute.gpus,
            gpus=cfg.compute.gpus,
            ram=cfg.compute.ram,
            time=cfg.compute.time,
            partition=cfg.server.partition,
            constraint=cfg.server.constraint,
            exclude="workergpu156",
        ),
        name=f"training lsm {runid}",
        backend="slurm",
        interpreter=f"torchrun --nnodes 1 --nproc-per-node {cfg.compute.gpus} --standalone",
        env=[
            "export OMP_NUM_THREADS=" + f"{cfg.compute.cpus_per_gpu}",
            "export WANDB_SILENT=true",
            "export XDG_CACHE_HOME=$HOME/.cache",
        ],
    )
