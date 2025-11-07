#!/usr/bin/env python

import argparse
from posixpath import split
import dawgz
import wandb

from functools import partial
from omegaconf import DictConfig

from lola.hydra import compose
from lola.latent3d import LatentVolumeFolder
from lola.lola.data import get_well_inputs

def train(runid: str, cfg: DictConfig):
    import os
    import re
    import torch
    import torch.distributed as dist
    import wandb

    from einops import rearrange
    from omegaconf import OmegaConf, open_dict
    from pathlib import Path
    from torch.nn.parallel import DistributedDataParallel
    from tqdm import trange

    from lola.data import (
        MiniWellDataset,
        field_preprocess,
        find_hdf5,
        get_dataloader,
       # get_well_inputs,
        get_well_multi_dataset,
    )
    from lola.latent3d import LatentVolumeFolder
    from lola.diffusion import DenoiserLoss, get_denoiser
    from lola.emulation import random_context_mask
    from lola.nn.utils import load_state_dict
    from lola.optim import get_optimizer, safe_gd_step
    from lola.utils import randseed

    # DDP
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device_id = os.environ.get("LOCAL_RANK", rank)
    device_id = int(device_id)
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)

    # Performance
    torch.set_float32_matmul_precision("high")

    # Config
    assert cfg.train.epoch_size % cfg.train.batch_size == 0
    assert cfg.train.batch_size % (cfg.train.accumulation * world_size) == 0
    assert cfg.valid.epoch_size % cfg.valid.batch_size == 0
    assert cfg.valid.batch_size % world_size == 0
    '''
    if cfg.ae_run:
        space = re.search(r"f\d+c\d+", cfg.ae_run).group()
    else:
        space = "pixel"
    '''
    if cfg.get("dataset",{}).get("type", "") == "latents":
        space = "latents"
    elif cfg.ae_run:
        space = re.search(r"f\d+c\d+", cfg.ae_run).group()
    else:
        space = "pixel"

    runname = f"{runid}_{cfg.dataset.name}_{space}_{cfg.denoiser.name}"

    runpath = Path(f"{cfg.server.storage}/runs/dm/{runname}")
    runpath = runpath.expanduser().resolve()
    runpath.mkdir(parents=True, exist_ok=True)

    with open_dict(cfg):
        cfg.name = runname
        cfg.path = str(runpath)
        cfg.seed = randseed(runid)

        if cfg.ae_run:
            cfg.ae_run = os.path.realpath(os.path.expanduser(cfg.ae_run), strict=True)

    if rank == 0 and cfg.ae_run:
        os.symlink(cfg.ae_run, runpath / "autoencoder")

    dist.barrier(device_ids=[device_id])

    # Data
    if cfg.get("dataset",{}).get("type", "") == "latents":
        train_root = getattr(cfg.dataset, "path", None)
        valid_root = getattr(cfg.dataset, "valid_path", None) or train_root
        if train_root is None:
            raise ValueError("For 'latents' dataset type, 'dataset.path' must be specified in the config.")
        dataset = {
            split: LatentVolumeFolder(
                root=train_root if split == "train" else valid_root,
                latent_key=getattr(cfg.dataset, "latent_key", "z_q"),
                channels=getattr(cfg.dataset, "channels", 512),
                normalize_mean=getattr(cfg.dataset, "normalize_mean", None),
                normalize_std=getattr(cfg.dataset, "normalize_std", None),
                max_samples=getattr(cfg.dataset, "max_samples", None),
            )
            for split in ("train", "valid")
        }
         # Adapter: dataset returns {"image": (B,C,D,H,W)} after collation.
        def get_latent_inputs(batch, device=None):
            x=batch["image"]
            if device is not None:
                x = x.to(device, non_blocking=True)
                # (B,C,D,H,W) -> (B,D,H,W,C) so the trainer's rearrange works
                x = x.movedim(1, -1).contiguous()
                B = x.shape[0]
                label = torch.empty(B, 0, device=x.device)  # unconditional
                return x, label
        
        preprocess = lambda x: x  # latents already normalized upstream

    elif cfg.ae_run:
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

        dataset = {
            split: MiniWellDataset.from_files(
                files=files[split],
                steps=cfg.trajectory.length,
                stride=cfg.trajectory.stride,
            )
            for split in ("train", "valid")
        }
    else:
        dataset = {
            split: get_well_multi_dataset(
                path=cfg.server.datasets,
                physics=cfg.dataset.physics,
                split=split,
                steps=cfg.trajectory.length,
                min_dt_stride=cfg.trajectory.stride,
                max_dt_stride=cfg.trajectory.stride,
                include_filters=cfg.dataset.include_filters,
                augment=cfg.dataset.augment,
            )
            for split in ("train", "valid")
        }

    train_loader, valid_loader = [
        get_dataloader(
            dataset=dataset[split],
            batch_size=(
                cfg.train.batch_size // cfg.train.accumulation // world_size
                if split == "train"
                else cfg.valid.batch_size // world_size
            ),
            shuffle=True,
            infinite=True,
            num_workers=cfg.compute.cpus_per_gpu,
            rank=rank,
            world_size=world_size,
            seed=cfg.seed,
        )
        for split in ("train", "valid")
    ]

    if cfg.get("dataset", {}).get("type", "") == "latents":
         # no extra preprocess; already set above
          pass
    elif cfg.ae_run:
        preprocess = lambda x: x
    else:
        preprocess = partial(
            field_preprocess,
            mean=torch.as_tensor(cfg.dataset.stats.mean, device=device),
            std=torch.as_tensor(cfg.dataset.stats.std, device=device),
            transform=cfg.dataset.transform,
        )

    #x, label = get_well_inputs(next(valid_loader))
    #x = rearrange(x, "B L ... C -> B C L ...")
    # Prime shapes for model config
    if cfg.get("dataset", {}).get("type", "") == "latents":
        batch0 = next(valid_loader)
        x, label = get_latent_inputs(batch0, device=device)
        x = preprocess(x)
        x = rearrange(x, "B L ... C -> B C L ...")  # L := D (depth)
    else:
        x, label = get_well_inputs(next(valid_loader))
        x = rearrange(x, "B L ... C -> B C L ...")


    # Model, optimizer & scheduler
    with open_dict(cfg):
        cfg.denoiser.channels = x.shape[1]
        #cfg.denoiser.label_features = label.shape[1]
        #cfg.denoiser.spatial = len(cfg.dataset.dimensions) + 1
        #cfg.denoiser.masked = True
        cfg.denoiser.spatial = len(cfg.dataset.dimensions) + 1  # +1 accounts for L=D
        # Unconditional: do NOT force labels/masks here
        # (leave masked/label_features to YAML; for unconditional set masked:false, label_features:0)

    denoiser = get_denoiser(**cfg.denoiser).to(device)
    denoiser_loss = DenoiserLoss(**cfg.denoiser.loss).to(device)

    if cfg.fork.run:
        stem_path = Path(cfg.fork.run).expanduser().resolve()
        stem_state = torch.load(stem_path / f"{cfg.fork.target}.pth", weights_only=True, map_location=device)

        load_state_dict(denoiser, stem_state, strict=cfg.fork.strict)

        del stem_state

    denoiser = DistributedDataParallel(
        module=denoiser,
        device_ids=[device_id],
    )

    optimizer, scheduler = get_optimizer(
        params=denoiser.parameters(),
        epochs=cfg.train.epochs,
        **cfg.optim,
    )

    # W&B
    if rank == 0:
        OmegaConf.save(cfg, runpath / "config.yaml")

    if rank == 0:
        run = wandb.init(
            entity=cfg.wandb.entity,
            project="lola-dm",
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

    for epoch in epochs:
        ## Train
        denoiser.train()

        losses, grads = [], []

        for i in range(cfg.train.accumulation * cfg.train.epoch_size // cfg.train.batch_size):
            #x, label = get_well_inputs(next(train_loader), device=device)
            #x = preprocess(x)
            #x = rearrange(x, "B L ... C -> B C L ...")

            #mask = random_context_mask(x, **cfg.trajectory.context)
            if cfg.get("dataset",{}).get("type", "") == "latents":
                batch = next(train_loader)
                x, label = get_latent_inputs(batch, device=device)
            else:
                x, label = get_well_inputs(next(train_loader), device=device)
            
            x = preprocess(x)
            x = rearrange(x, "B L ... C -> B C L ...")


            if (i + 1) % cfg.train.accumulation == 0:
                #loss = denoiser_loss(denoiser, x, mask=mask, label=label)
                 # Unconditional: no mask, no label
                loss = denoiser_loss(denoiser, x)
                loss_acc = loss / cfg.train.accumulation
                loss_acc.backward()

                grad_norm = safe_gd_step(optimizer, grad_clip=cfg.optim.grad_clip)
                grads.append(grad_norm)
            else:
                with denoiser.no_sync():
                    #loss = denoiser_loss(denoiser, x, mask=mask, label=label)
                    loss = denoiser_loss(denoiser, x)
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

        del losses, losses_list, grads, grads_list

        ## Eval
        denoiser.eval()

        losses = []

        with torch.no_grad():
            for _ in range(cfg.valid.epoch_size // cfg.valid.batch_size):
               # x, label = get_well_inputs(next(valid_loader), device=device)
               # x = preprocess(x)
               # x = rearrange(x, "B L ... C -> B C L ...")

                #mask = random_context_mask(x, **cfg.trajectory.context)

                #loss = denoiser_loss(denoiser, x, mask=mask, label=label)
                #losses.append(loss)
                if cfg.get("dataset",{}).get("type", "") == "latents":
                    batch = next(valid_loader)
                    x, label = get_latent_inputs(batch, device=device)
                else:
                    x, label = get_well_inputs(next(valid_loader), device=device)
                x = preprocess(x)
                x = rearrange(x, "B L ... C -> B C L ...")
                loss = denoiser_loss(denoiser, x)
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

            run.log(logs, step=epoch)

        del losses, losses_list

        ## LR scheduler
        scheduler.step()

        ## Checkpoint
        if rank == 0:
            state = denoiser.module.state_dict()

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
        config_file="./configs/train_diffusion.yaml",
        overrides=args.overrides,
    )

    # Job
    runid = wandb.util.generate_id()

    if cfg.compute.nodes > 1:
        interpreter = f"torchrun --nnodes {cfg.compute.nodes} --nproc-per-node {cfg.compute.gpus} --rdzv_backend=c10d --rdzv_endpoint=$SLURMD_NODENAME:12345 --rdzv_id=$SLURM_JOB_ID"
    else:
        interpreter = f"torchrun --nnodes 1 --nproc-per-node {cfg.compute.gpus} --standalone"

    dawgz.schedule(
        dawgz.job(
            f=partial(train, runid, cfg),
            name=f"dm {runid}",
            nodes=cfg.compute.nodes,
            cpus=cfg.compute.cpus_per_gpu * cfg.compute.gpus,
            gpus=cfg.compute.gpus,
            ram=cfg.compute.ram,
            time=cfg.compute.time,
            partition=cfg.server.partition,
            constraint=cfg.server.constraint,
            exclude=cfg.server.exclude,
        ),
        name=f"training dm {runid}",
        backend="slurm",
        interpreter=interpreter,
        env=[
            "export OMP_NUM_THREADS=" + f"{cfg.compute.cpus_per_gpu}",
            "export WANDB_SILENT=true",
            "export XDG_CACHE_HOME=$HOME/.cache",
        ],
    )
