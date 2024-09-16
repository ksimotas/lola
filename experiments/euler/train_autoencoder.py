#!/usr/bin/env python

import argparse
import os

from dawgz import job, schedule
from lpdm.hydra import multi_compose
from omegaconf import DictConfig


def train_ae(
    cfg: DictConfig,
    datasets: str = "/mnt/ceph/users/polymathic/the_well/datasets",
    device: str = "cuda",
):
    import neptune
    import torch
    import uuid

    from einops import rearrange
    from functools import partial
    from itertools import islice
    from lpdm.data import field_preprocess, get_well_dataset
    from lpdm.nn.autoencoder import AutoEncoder
    from lpdm.optim import get_optimizer, safe_gd_step
    from neptune.utils import stringify_unsupported
    from omegaconf import OmegaConf, open_dict
    from pathlib import Path
    from torch.utils.data import DataLoader
    from tqdm import trange

    # Config
    with open_dict(cfg):
        cfg.uuid = uuid.uuid4().hex
        cfg.name = f"{cfg.dataset.name}_{cfg.ae.name}_{cfg.optim.name}"

    runpath = Path("~/ceph") / "mpp-ldm" / "runs" / f"{cfg.name}_{cfg.uuid}"
    runpath = runpath.expanduser().resolve()
    runpath.mkdir(parents=True, exist_ok=True)

    with open_dict(cfg):
        cfg.path = str(runpath)

    with open(runpath / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Data
    trainset = get_well_dataset(
        path=os.path.join(datasets, cfg.dataset.physics, "data/train"),
        in_steps=1,
        include_filters=cfg.dataset.include_filters,
    )

    validset = get_well_dataset(
        path=os.path.join(datasets, cfg.dataset.physics, "data/valid"),
        in_steps=1,
        include_filters=cfg.dataset.include_filters,
    )

    train_loader = DataLoader(
        dataset=trainset,
        shuffle=True,
        batch_size=cfg.train.batch_size,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        dataset=validset,
        shuffle=True,
        batch_size=cfg.train.batch_size,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
    )

    preprocess = partial(
        field_preprocess,
        mean=torch.as_tensor(cfg.dataset.stats.mean, device=device),
        std=torch.as_tensor(cfg.dataset.stats.std, device=device),
        transform=cfg.dataset.transform,
    )

    # Model
    model = AutoEncoder(
        pix_channels=len(cfg.dataset.fields),
        lat_channels=cfg.ae.lat_channels,
        hid_channels=cfg.ae.hid_channels,
        hid_blocks=cfg.ae.hid_blocks,
        dropout=0.01,
        spatial=2,
        periodic=cfg.dataset.periodic,
        checkpointing=True,
    )

    model.to(device)

    # Optimizer & Scheduler
    optimizer, scheduler = get_optimizer(
        params=model.parameters(),
        optimizer=cfg.optim.optimizer,
        learning_rate=cfg.optim.learning_rate,
        weight_decay=cfg.optim.weight_decay,
        scheduler=cfg.optim.scheduler,
        epochs=cfg.train.epochs,
    )

    # Neptune
    run = neptune.init_run(
        project="mppx/mpp-ldm",
        name=cfg.name,
        source_files=["train.py"],
    )

    run["config"] = stringify_unsupported(OmegaConf.to_container(cfg))
    run["config/yaml"] = OmegaConf.to_yaml(cfg)

    # Training loop
    for epoch in trange(cfg.train.epochs, ncols=88):  # noqa: B007
        ## Train
        model.train()

        losses, grads = [], []

        for batch in islice(train_loader, cfg.train.epoch_size // cfg.train.batch_size):
            x = batch["input_fields"]
            x = preprocess(x)
            x = rearrange(x, "... B 1 H W C -> ... B C H W")
            x = x.to(device, non_blocking=True)

            loss = model.loss(x)
            loss.backward()

            grad_norm = safe_gd_step(optimizer, grad_clip=cfg.optim.grad_clip)

            losses.append(loss.detach())
            grads.append(grad_norm)

        losses, grads = torch.stack(losses).cpu(), torch.stack(grads).cpu()

        run["train/loss/mean"].append(losses.mean())
        run["train/loss/std"].append(losses.std())
        run["train/grad_norm/mean"].append(grads.mean())
        run["train/grad_norm/std"].append(grads.std())

        ## Eval
        model.eval()

        losses = []

        with torch.no_grad():
            for batch in islice(valid_loader, cfg.train.epoch_size // cfg.train.batch_size):
                x = batch["input_fields"]
                x = preprocess(x)
                x = rearrange(x, "... B 1 H W C -> ... B C H W")
                x = x.to(device, non_blocking=True)

                loss = model.loss(x).detach()
                losses.append(loss)

        losses = torch.stack(losses).cpu()

        run["valid/loss/mean"].append(losses.mean())
        run["valid/loss/std"].append(losses.std())

        ## LR scheduler
        scheduler.step()

        ## Checkpoint
        state = model.state_dict()
        torch.save(state, runpath / "state.pth")

    run.stop()


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpuxl", action="store_true", default=False)
    parser.add_argument("--slurm", action="store_true", default=False)

    args = parser.parse_args()

    # Config(s)
    configs = multi_compose(
        config_dir="./configs",
        config_name="default.yaml",
        overrides=args.overrides,
    )

    if args.gpuxl:
        datasets = "/mnt/gpuxl/polymathic/the_well/datasets"
    else:
        datasets = "/mnt/ceph/users/polymathic/the_well/datasets"

    def run(i: int):
        train_ae(
            configs[i],
            datasets=datasets,
            device=args.device,
        )

    # Run
    if args.slurm:
        schedule(
            job(
                f=run,
                name="train_ae",
                array=len(configs),
                cpus=16,
                gpus=1,
                ram="64GB",
                time="1-00:00:00",
                partition="gpu",
                constraint="h100",
            ),
            backend="slurm",
        )
    else:
        for i in range(len(configs)):
            if i > 0:
                print("-" * 88)
            run(i)
