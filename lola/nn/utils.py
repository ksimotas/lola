r"""Miscellaneous module helpers."""

import torch.nn as nn

from torch import Tensor
from typing import Dict, Sequence, Tuple


def load_state_dict(
    self: nn.Module,
    state_dict: Dict[str, Tensor],
    strict: bool = False,
    translate: Sequence[Tuple[str, str]] = (),
) -> nn.Module:
    for a, b in translate:
        state_dict = {key.replace(a, b): value for key, value in state_dict.items()}

    if strict:
        return self.load_state_dict(state_dict)

    current = self.state_dict()

    for key in current:
        if key in state_dict:
            if state_dict[key].shape == current[key].shape:
                current[key] = state_dict[key]

    return self.load_state_dict(current)
