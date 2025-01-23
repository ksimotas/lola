r"""Miscellaneous module helpers."""

import torch.nn as nn

from torch import Tensor
from typing import Dict


def load_state_dict(
    self: nn.Module,
    state_dict: Dict[str, Tensor],
    strict: bool = False,
):
    if strict:
        return self.load_state_dict(state_dict)

    current = self.state_dict()

    for key in current:
        if key in state_dict:
            if state_dict[key].shape == current[key].shape:
                current[key] = state_dict[key]

    return self.load_state_dict(current)
