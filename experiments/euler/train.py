r"""Training auto-encoders."""

from dawgz import Job, schedule
from lpdm.hydra import multi_compose
from omegaconf import DictConfig, OmegaConf, open_dict
from pathlib import Path


def train(cfg: DictConfig):
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
    from torch.utils.data import DataLoader
    from tqdm import trange

    # Config
    with open_dict(cfg):
        cfg.uuid = uuid.uuid4().hex
        cfg.name = f"{cfg.dataset.name}_{cfg.ae.name}"

    runpath = Path("~/ceph") / "mpp-ldm" / "runs" / f"{cfg.name}_{cfg.uuid}"
    runpath = runpath.expanduser().resolve()
    runpath.mkdir(parents=True, exist_ok=True)

    with open_dict(cfg):
        cfg.path = str(runpath)

    with open(runpath / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Device
    device = "cuda"

    # Data
    trainset = get_well_dataset(
        path=f"/mnt/home/polymathic/ceph/the_well/datasets/{cfg.dataset.physics}/data/train",
        in_steps=1,
        include_filters=cfg.dataset.include_filters,
    )

    validset = get_well_dataset(
        path=f"/mnt/home/polymathic/ceph/the_well/datasets/{cfg.dataset.physics}/data/valid",
        in_steps=1,
        include_filters=cfg.dataset.include_filters,
    )

    train_loader = DataLoader(
        dataset=trainset,
        shuffle=True,
        batch_size=cfg.trainer.batch_size,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        dataset=validset,
        shuffle=True,
        batch_size=cfg.trainer.batch_size,
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
        dropout=0.1,
        spatial=2,
    )

    model.to(device)

    # Optimizer & Scheduler
    optimizer, scheduler = get_optimizer(
        params=model.parameters(),
        optimizer=cfg.trainer.optimizer,
        learning_rate=cfg.trainer.learning_rate,
        weight_decay=cfg.trainer.weight_decay,
        scheduler=cfg.trainer.scheduler,
        epochs=cfg.trainer.epochs,
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
    for epoch in trange(cfg.trainer.epochs, ncols=88):  # noqa: B007
        ## Train
        model.train()

        losses, grads = [], []

        for batch in islice(train_loader, cfg.trainer.epoch_size // cfg.trainer.batch_size):
            x = batch["input_fields"]
            x = preprocess(x)
            x = rearrange(x, "... B 1 H W C -> ... B C H W")
            x = x.to(device, non_blocking=True)

            loss = model.loss(x)
            loss, grad_norm = safe_gd_step(loss, optimizer, grad_clip=cfg.trainer.grad_clip)

            losses.append(loss)
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
            for batch in islice(valid_loader, cfg.trainer.epoch_size // cfg.trainer.batch_size):
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
    configs = multi_compose(
        config_dir="./configs",
        config_name="default",
        overrides=[
            "ae=64x64_small,128x128_small",
        ],
    )

    def main(i: int):
        train(configs[i])

    job = Job(
        f=main,
        name="train",
        array=len(configs),
        time="1-00:00:00",
        cpus=16,
        gpus=1,
        ram="64GB",
        partition="gpu",
        constraint="h100",
    )

    schedule(
        job,
        name="training auto-encoders",
        backend="slurm",
        export="ALL",
    )
