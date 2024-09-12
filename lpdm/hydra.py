r"""Hydra helpers."""

import hydra
import itertools
import os

from omegaconf import DictConfig
from typing import List, Sequence


def multi_compose(
    config_dir: str,
    config_name: str,
    overrides: Sequence[str] = (),
) -> List[DictConfig]:
    r"""Composes Hydra configurations.

    Arguments:
        config_dir: The configuration directory.
        config_name: The configuration file name (with or without .yaml extension).
        overrides: The overriden parameters, as a sequence of strings :py:`"key=v0,v1,v2"`.

    Returns:
        A list of configurations.
    """

    keys, grid = [], []

    for line in overrides:
        key, values = line.split("=")
        keys.append(key)
        grid.append(values.split(","))

    cfgs = []

    with hydra.initialize_config_dir(config_dir=os.path.abspath(config_dir), version_base=None):
        for values in itertools.product(*grid):
            overrides = [f"{key}={val}" for key, val in zip(keys, values, strict=True)]
            cfg = hydra.compose(config_name=config_name, overrides=overrides)
            cfgs.append(cfg)

    return cfgs
