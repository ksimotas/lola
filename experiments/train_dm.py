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

    from lpdm.data import field_preprocess, get_dataloader, get_label, get_well_multi_dataset
    from lpdm.diffusion import DenoiserLoss, get_denoiser
    from lpdm.optim import ExponentialMovingAverage, get_optimizer, safe_gd_step
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

    runname = f"{cfg.denoiser.name}_{cfg.optim.name}"

    runpath = Path(f"~/ceph/mpp-ldm/runs/{runid}_{runname}")
    runpath = runpath.expanduser().resolve()
    runpath.mkdir(parents=True, exist_ok=True)

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
        stem_dir = os.path.basename(stem.config["path"])
        stem_path = Path(f"~/ceph/mpp-ldm/runs/{stem_dir}")
        stem_path = stem_path.expanduser().resolve()
        stem_state = torch.load(stem_path / f"{cfg.fork_target}.pth", weights_only=True)

        counter = {
            "epoch": stem.summary["_step"] + 1,
            "update_samples": stem.summary["train/samples"],
            "update_steps": stem.summary["train/update_steps"],
        }

    # Data
    trainset = get_well_multi_dataset(
        path=cfg.server.datasets,
        physics=cfg.dataset.physics,
        split="train",
        steps=cfg.trajectory.length,
        min_dt_stride=cfg.trajectory.stride,
        max_dt_stride=cfg.trajectory.stride,
        include_filters=cfg.dataset.include_filters,
        augment=cfg.dataset.augment,
    )

    train_loader = get_dataloader(
        dataset=trainset,
        batch_size=cfg.train.batch_size // world_size,
        shuffle=True,
        infinite=True,
        num_workers=process_cpu_count() // world_size,
        rank=rank,
        world_size=world_size,
        seed=cfg.seed,
    )

    validset = get_well_multi_dataset(
        path=cfg.server.datasets,
        physics=cfg.dataset.physics,
        split="valid",
        steps=cfg.trajectory.length,
        min_dt_stride=cfg.trajectory.stride,
        max_dt_stride=cfg.trajectory.stride,
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

    batch = next(valid_loader)
    batch["input_fields"] = rearrange(batch["input_fields"], "B L H W C -> B C L H W")
    batch["label"] = get_label(batch)

    # Model, optimizer & scheduler
    denoiser = get_denoiser(
        shape=batch["input_fields"].shape[1:],
        label_features=batch["label"].shape[1],
        **cfg.denoiser,
    ).to(device)

    denoiser_loss = DenoiserLoss(**cfg.denoiser.loss).to(device)

    if cfg.fork_from is not None:
        denoiser.load_state_dict(stem_state)

    denoiser = DistributedDataParallel(
        module=denoiser,
        device_ids=[device_id],
    )

    optimizer, scheduler = get_optimizer(
        params=denoiser.parameters(),
        epochs=cfg.train.epochs,
        **cfg.optim,
    )

    average = ExponentialMovingAverage(
        module=denoiser.module,
        decay=cfg.train.ema_decay,
    )

    # W&B
    if rank == 0:
        run = wandb.init(
            project="mpp-dm",
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
        denoiser.train()

        losses, grads = [], []

        for i in range(cfg.train.epoch_size // cfg.train.batch_size):
            batch = next(train_loader)

            x = batch["input_fields"]
            x = x.to(device, non_blocking=True)
            x = preprocess(x)
            x = rearrange(x, "B L H W C -> B C L H W")

            if cfg.masking:
                mask = torch.rand(
                    (x.shape[0], 1, x.shape[2], 1, 1),
                    dtype=x.dtype,
                    device=x.device,
                )
                mask = mask < 1 / x.shape[2]
                mask = mask.expand(x.shape).contiguous()
            else:
                mask = None

            label = get_label(batch)
            label = label.to(device, non_blocking=True)

            if (i + 1) % cfg.train.accumulation == 0:
                loss = denoiser_loss(denoiser, x, mask=mask, label=label)
                loss_acc = loss / cfg.train.accumulation
                loss_acc.backward()

                grad_norm = safe_gd_step(optimizer, grad_clip=cfg.optim.grad_clip)
                grads.append(grad_norm)

                average.update_parameters(denoiser.module)

                counter["update_samples"] += cfg.train.batch_size * cfg.train.accumulation
                counter["update_steps"] += 1
            else:
                with denoiser.no_sync():
                    loss = denoiser_loss(denoiser, x, mask=mask, label=label)
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
        denoiser.eval()

        losses = []

        with torch.no_grad():
            for _ in range(cfg.train.epoch_size // cfg.train.batch_size):
                batch = next(valid_loader)

                x = batch["input_fields"]
                x = x.to(device, non_blocking=True)
                x = preprocess(x)
                x = rearrange(x, "B L H W C -> B C L H W")

                if cfg.masking:
                    mask = torch.rand(
                        (x.shape[0], 1, x.shape[2], 1, 1),
                        dtype=x.dtype,
                        device=x.device,
                    )
                    mask = mask < 1 / x.shape[2]
                    mask = mask.expand(x.shape).contiguous()
                else:
                    mask = None

                label = get_label(batch)
                label = label.to(device, non_blocking=True)

                loss = denoiser_loss(denoiser, x, mask=mask, label=label)
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
            state = denoiser.module.state_dict()
            state_ema = average.module.state_dict()

            torch.save(state, runpath / "state.pth")
            torch.save(state_ema, runpath / "state_ema.pth")

            if logs["valid/loss/mean"] < best_valid_loss:
                best_valid_loss = logs["valid/loss/mean"]

                torch.save(state, runpath / "state_best.pth")
                torch.save(state_ema, runpath / "state_best_ema.pth")

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
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--ram", type=str, default="512GB")
    parser.add_argument("--time", type=str, default="7-00:00:00")

    args = parser.parse_args()

    # Config
    cfg = compose(
        config_file="./configs/default_dm.yaml",
        overrides=args.overrides,
    )

    # Job
    runid = wandb.util.generate_id()

    dawgz.schedule(
        dawgz.job(
            f=partial(train, runid, cfg),
            name=f"dm {runid}",
            cpus=args.cpus_per_gpu * args.gpus,
            gpus=args.gpus,
            ram=args.ram,
            time=args.time,
            partition=cfg.server.partition,
            constraint=cfg.server.constraint,
        ),
        name=f"training dm {runid}",
        backend="slurm",
        interpreter=f"torchrun --nnodes 1 --nproc-per-node {args.gpus} --standalone",
        env=[
            "export OMP_NUM_THREADS=" + f"{args.cpus_per_gpu}",
            "export WANDB_SILENT=true",
            "export XDG_CACHE_HOME=$HOME/.cache",
        ],
    )
