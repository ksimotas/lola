r"""Tests for the lpdm.nn.dit module."""

import pytest
import torch

from pathlib import Path

from lpdm.nn.dit import DiT


@pytest.mark.parametrize("length", [16])
@pytest.mark.parametrize("in_channels, out_channels", [(3, 5)])
@pytest.mark.parametrize("mod_features", [16])
@pytest.mark.parametrize("attention_heads", [1, 4])
@pytest.mark.parametrize("dropout", [None, 0.1])
@pytest.mark.parametrize("spatial", [1, 2])
@pytest.mark.parametrize("patch_size", [2, 4])
@pytest.mark.parametrize("window_size", [None, 3])
@pytest.mark.parametrize("batch_size", [4])
def test_DiT(
    tmp_path: Path,
    length: int,
    in_channels: int,
    out_channels: int,
    mod_features: int,
    attention_heads: int,
    dropout: float,
    spatial: int,
    patch_size: int,
    window_size: int,
    batch_size: int,
):
    make = lambda: DiT(
        in_channels=in_channels,
        out_channels=out_channels,
        mod_features=mod_features,
        hid_channels=64,
        hid_blocks=3,
        attention_heads=attention_heads,
        dropout=dropout,
        spatial=spatial,
        patch_size=patch_size,
        window_size=None if window_size is None else (window_size,) * spatial,
    )

    net = make()
    net.train()

    # Call
    x = torch.randn((batch_size, in_channels) + (length,) * spatial)
    mod = torch.randn(batch_size, mod_features)
    y = net(x, mod)

    assert y.ndim == x.ndim
    assert y.shape[0] == batch_size
    assert y.shape[1] == out_channels
    assert y.shape[2:] == x.shape[2:]

    ## Grads
    assert y.requires_grad

    loss = y.square().sum()
    loss.backward()

    for p in net.parameters():
        assert p.grad is not None
        assert torch.all(torch.isfinite(p.grad))

    # Save
    torch.save(net.state_dict(), tmp_path / "state.pth")

    # Load
    copy = make()
    copy.load_state_dict(torch.load(tmp_path / "state.pth", weights_only=True))

    net.eval()
    copy.eval()

    y_net = net(x, mod)
    y_copy = copy(x, mod)

    assert torch.allclose(y_net, y_copy)
