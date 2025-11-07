#!/usr/bin/env python
import argparse
from functools import partial
from pathlib import Path

import dawgz
import wandb

from omegaconf import DictConfig
from lola.hydra import compose

def train(runid: str, cfg: DictConfig):
    import os
    import torch
    import torch.distributed as dist

    from einops import rearrange
    from omegaconf import OmegaConf, open_dict
    from pathlib import Path
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler

    from lola.latent3d import LatentVolumeFolder
    from lola.diffusion import DenoiserLoss, get_denoiser
    from lola.nn.utils import load_state_dict
    from lola.optim import get_optimizer, safe_gd_step
    from lola.utils import randseed
    from tqdm import trange

    # ---------------------------
    # DDP init
    # ---------------------------
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device_id = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)

    # Perf
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    # ---------------------------
    # Config sanity checks
    # ---------------------------
    assert cfg.train.epoch_size % cfg.train.batch_size == 0
    assert cfg.train.batch_size % (cfg.train.accumulation * world_size) == 0
    assert cfg.valid.epoch_size % cfg.valid.batch_size == 0
    assert cfg.valid.batch_size % world_size == 0

    # Run naming & paths
    ds_name = getattr(cfg.dataset, "name", Path(getattr(cfg.dataset, "path", "latents")).stem) or "latents"
    runname = f"{runid}_{ds_name}_latents_{cfg.denoiser.name}"

    runpath = Path(f"{cfg.server.storage}/runs/dm/{runname}").expanduser().resolve()
    runpath.mkdir(parents=True, exist_ok=True)

    with open_dict(cfg):
        cfg.name = runname
        cfg.path = str(runpath)
        cfg.seed = randseed(runid)

    # Seed torch for reproducibility across ranks
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    dist.barrier()  # plain barrier


    #OPTIONAL: load per-channel stats from file if provided (DDP-safe)
    stats = None
    stats_file = getattr(cfg.dataset, "stats_file", None)
    if stats_file:
        sf = os.path.expanduser(os.path.realpath(stats_file))
        if rank == 0:
            import torch as _torch
            sdict = _torch.load(sf, map_location="cpu")
            stats = {
                "mean": _torch.as_tensor(sdict.get("mean")),
                "std":  _torch.as_tensor(sdict.get("std")),
            }
        # broadcast to all ranks
        obj_list = [stats]
        dist.broadcast_object_list(obj_list, src=0)
        stats = obj_list[0]

    from omegaconf import open_dict as _open_dict
    with _open_dict(cfg):
        cfg.dataset.normalize_mean = stats["mean"].tolist()
        cfg.dataset.normalize_std  = stats["std"].tolist()

    # ---------------------------
    # Dataset: latent volumes only
    # ---------------------------
    train_root = getattr(cfg.dataset, "path", None)
    valid_root = getattr(cfg.dataset, "valid_path", None) or train_root
    if train_root is None:
        raise ValueError("cfg.dataset.path must be set for latent training.")
    
    train_set = LatentVolumeFolder(
        root=train_root,
        latent_key=getattr(cfg.dataset, "latent_key", "z_q"),
        channels=getattr(cfg.dataset, "channels", 512),
        item_index=getattr(cfg.dataset, "item_index", 0),
        normalize_mean=getattr(cfg.dataset, "normalize_mean", None),
        normalize_std=getattr(cfg.dataset, "normalize_std", None),
        max_samples=getattr(cfg.dataset, "max_samples", None),
    )

    valid_set = LatentVolumeFolder(
        root=valid_root,
        latent_key=getattr(cfg.dataset, "latent_key", "z_q"),
        channels=getattr(cfg.dataset, "channels", 512),
        item_index=getattr(cfg.dataset, "val_item_index", getattr(cfg.dataset, "item_index", 0)),
        normalize_mean=getattr(cfg.dataset, "normalize_mean", None),
        normalize_std=getattr(cfg.dataset, "normalize_std", None),
        max_samples=getattr(cfg.dataset, "val_max_samples", None),
    )

    def make_loader(ds, global_bs, shuffle):
        sampler = DistributedSampler(
            ds, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=True
        )
        return DataLoader(
            ds,
            batch_size=global_bs // world_size,
            sampler=sampler,
            num_workers=cfg.compute.cpus_per_gpu,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(cfg.compute.cpus_per_gpu > 0),
        )
    
    train_loader = make_loader(train_set, cfg.train.batch_size, shuffle=True)
    valid_loader = make_loader(valid_set, cfg.valid.batch_size, shuffle=False)

    # Adapter: dataset returns {"image": (B,C,D,H,W)}
    def get_latent_inputs(batch, device=None):
        x = batch["image"]  # (B, C, D, H, W)
        if device is not None:
            x = x.to(device, non_blocking=True)
        x = x.movedim(1, -1).contiguous()  # -> (B, D, H, W, C)
        B = x.shape[0]
        label = torch.empty(B, 0, device=x.device)  # unconditional
        return x, label
    
    preprocess = lambda x: x  # already normalized if mean/std provided

    # Prime shapes & set denoiser config
    batch0 = next(iter(valid_loader))
    x, _ = get_latent_inputs(batch0, device=device)
    x = preprocess(x)
    x = rearrange(x, "B D H W C -> B C D H W")

    with open_dict(cfg):
        cfg.denoiser.channels = x.shape[1]
        # (#spatial dims) = x.ndim - 2 for (B, C, ..., ...)
        cfg.denoiser.spatial = x.ndim - 2

    # ---------------------------
    # Model, optimizer, sched
    # ---------------------------
    denoiser = get_denoiser(**cfg.denoiser).to(device)
    denoiser_loss = DenoiserLoss(**cfg.denoiser.loss).to(device)

    if getattr(cfg.fork, "run", None):
        stem_path = Path(cfg.fork.run).expanduser().resolve()
        stem_state = torch.load(stem_path / f"{cfg.fork.target}.pth", weights_only=True, map_location=device)
        load_state_dict(denoiser, stem_state, strict=getattr(cfg.fork, "strict", False))
        del stem_state

    denoiser = DistributedDataParallel(denoiser, device_ids=[device_id])

    optimizer, scheduler = get_optimizer(
        params=denoiser.parameters(),
        epochs=cfg.train.epochs,
        **cfg.optim,
    )

    # ---------------------------
    # Weights & Biases
    # ---------------------------
    if rank == 0:
        OmegaConf.save(cfg, runpath / "config.yaml")
        run = wandb.init(
            entity=cfg.wandb.entity,
            project="lola-dm",
            id=runid,
            name=runname,
            config=OmegaConf.to_container(cfg),
        )

    # Helper to recycle an iterator
    def next_or_cycle(it, loader):
        try:
            return next(it), it
        except StopIteration:
            it = iter(loader)
            return next(it), it

    # ---------------------------
    # Training / Eval
    # ---------------------------
    epochs_iter = trange(cfg.train.epochs, ncols=88, ascii=True) if rank == 0 else range(cfg.train.epochs)
    best_valid_loss = float("inf")

    for epoch in epochs_iter:
        # per-epoch sampler seeds
        train_loader.sampler.set_epoch(epoch)
        valid_loader.sampler.set_epoch(epoch)

        # ---- Train ----
        denoiser.train()
        losses_epoch, grads_epoch = [], []

        train_it = iter(train_loader)
        num_steps = cfg.train.epoch_size // cfg.train.batch_size
        for step in range(num_steps):
            (batch, train_it) = next_or_cycle(train_it, train_loader)
            x, _ = get_latent_inputs(batch, device=device)
            x = preprocess(x)
            x = rearrange(x, "B D H W C -> B C D H W")

            # accumulation-aware sync
            if (step + 1) % cfg.train.accumulation == 0:
                loss = denoiser_loss(denoiser, x)
                (loss / cfg.train.accumulation).backward()
                grad_norm = safe_gd_step(optimizer, grad_clip=cfg.optim.grad_clip)
                grads_epoch.append(grad_norm)
            else:
                with denoiser.no_sync():
                    loss = denoiser_loss(denoiser, x)
                    (loss / cfg.train.accumulation).backward()

            losses_epoch.append(loss.detach())

        # gather train stats
        losses_t = torch.stack(losses_epoch)
        grads_t = torch.stack(grads_epoch) if len(grads_epoch) > 0 else torch.zeros(1, device=device)

        losses_list = [torch.empty_like(losses_t) for _ in range(world_size)]
        grads_list = [torch.empty_like(grads_t) for _ in range(world_size)]
        dist.all_gather(losses_list, losses_t)
        dist.all_gather(grads_list, grads_t)

        if rank == 0:
            losses_cat = torch.cat(losses_list).cpu()
            grads_cat = torch.cat(grads_list).cpu()
            logs = {
                "train/loss/mean": losses_cat.mean().item(),
                "train/loss/std": losses_cat.std(unbiased=False).item(),
                "train/grad_norm/mean": grads_cat.mean().item(),
                "train/grad_norm/std": grads_cat.std(unbiased=False).item(),
                "train/learning_rate": optimizer.param_groups[0]["lr"],
            }

        # ---- Eval ----
        denoiser.eval()
        valid_losses = []
        with torch.no_grad():
            valid_it = iter(valid_loader)
            valid_steps = cfg.valid.epoch_size // cfg.valid.batch_size
            for _ in range(valid_steps):
                (batch, valid_it) = next_or_cycle(valid_it, valid_loader)
                x, _ = get_latent_inputs(batch, device=device)
                x = preprocess(x)
                x = rearrange(x, "B D H W C -> B C D H W")
                loss = denoiser_loss(denoiser, x)
                valid_losses.append(loss)

        valid_t = torch.stack(valid_losses)
        valid_list = [torch.empty_like(valid_t) for _ in range(world_size)]
        dist.all_gather(valid_list, valid_t)

        if rank == 0:
            valid_cat = torch.cat(valid_list).cpu()
            logs["valid/loss/mean"] = valid_cat.mean().item()
            logs["valid/loss/std"] = valid_cat.std(unbiased=False).item()

            if hasattr(epochs_iter, "set_postfix"):
                epochs_iter.set_postfix(lt=logs["train/loss/mean"], lv=logs["valid/loss/mean"])

            run.log(logs, step=epoch)

        # LR step
        scheduler.step()

        # Checkpoint
        if rank == 0:
            state = denoiser.module.state_dict()
            torch.save(state, runpath / "state.pth")
            if logs["valid/loss/mean"] < best_valid_loss:
                best_valid_loss = logs["valid/loss/mean"]
                torch.save(state, runpath / "state_best.pth")
            del state

        dist.barrier()

    if rank == 0:
        run.finish()

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", type=str)
    args = parser.parse_args()

    script_root = Path(__file__).resolve().parent
    default_cfg = script_root / "configs" / "train_ldm512.yaml"

    if not default_cfg.is_file():
        raise FileNotFoundError(
            f"Expected default config at {default_cfg}, but it was not found. "
            "Update the path or provide overrides pointing to a valid config."
        )

    cfg = compose(
        config_file=str(default_cfg),
        overrides=args.overrides,
    )

    runid = wandb.util.generate_id()

    if cfg.compute.nodes > 1:
        interpreter = (
            f"torchrun --nnodes {cfg.compute.nodes} "
            f"--nproc-per-node {cfg.compute.gpus} "
            f"--rdzv_backend=c10d --rdzv_endpoint=$SLURMD_NODENAME:12345 "
            f"--rdzv_id=$SLURM_JOB_ID"
        )
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
