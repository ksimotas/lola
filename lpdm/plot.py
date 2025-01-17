r"""Plotting and image helpers."""

import math
import matplotlib.animation as ani
import matplotlib.pyplot as plt
import numpy as np
import torch

from numpy.typing import ArrayLike
from PIL import Image
from torch import Tensor
from typing import Optional, Sequence, Tuple


def animate_fields(
    x: ArrayLike,
    y: Optional[ArrayLike] = None,
    tolerance: float = 1.0,
    fields: Optional[Sequence[str]] = None,
    timesteps: Optional[Sequence[int]] = None,
    cmap: str = "RdBu_r",
    figsize: Tuple[float, float] = (2.4, 2.4),
    wait: float = 0.2,
) -> ani.Animation:
    C, L, _, _ = x.shape

    if timesteps is None:
        timesteps = list(range(L))

    if y is None:
        fig, axs = plt.subplots(
            nrows=1,
            ncols=C,
            figsize=(figsize[0] * C, figsize[1]),
            squeeze=False,
        )
    else:
        fig, axs = plt.subplots(
            nrows=3,
            ncols=C,
            figsize=(figsize[0] * C, 3 * figsize[1]),
            squeeze=False,
        )

    artists = []

    for i in range(C):
        vmin = np.nanmin(x[i])
        vmax = np.nanmax(x[i])

        if fields:
            axs[0, i].set_title(f"{fields[i]}")

        for j in range(1):
            if y is None:
                img = axs[j, i].imshow(
                    x[i, j], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none"
                )
                axs[j, i].set_xticks([])
                axs[j, i].set_yticks([])

                axs[j, 0].set_ylabel("$x_i$")

                artists.append(img)
            else:
                img0 = axs[3 * j, i].imshow(
                    x[i, j], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none"
                )
                axs[3 * j, i].set_xticks([])
                axs[3 * j, i].set_yticks([])

                img1 = axs[3 * j + 1, i].imshow(
                    y[i, j], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none"
                )
                axs[3 * j + 1, i].set_xticks([])
                axs[3 * j + 1, i].set_yticks([])

                img2 = axs[3 * j + 2, i].imshow(
                    y[i, j] - x[i, j],
                    cmap="RdBu_r",
                    vmin=-tolerance,
                    vmax=tolerance,
                    interpolation="none",
                )
                axs[3 * j + 2, i].set_xticks([])
                axs[3 * j + 2, i].set_yticks([])

                axs[3 * j, 0].set_ylabel("$x_i$")
                axs[3 * j + 1, 0].set_ylabel("$y_i$")
                axs[3 * j + 2, 0].set_ylabel("$y_i - x_i$")

                artists.extend((img0, img1, img2))

    def animate(j):
        for i in range(C):
            if y is None:
                artists[i].set_array(x[i, j])
            else:
                artists[3 * i + 0].set_array(x[i, j])
                artists[3 * i + 1].set_array(y[i, j])
                artists[3 * i + 2].set_array(y[i, j] - x[i, j])

        return artists

    fig.align_labels()
    fig.tight_layout()

    return ani.FuncAnimation(fig, animate, frames=L, interval=int(1000 * wait))


def draw_fields(
    x: ArrayLike,
    y: Optional[ArrayLike] = None,
    tolerance: float = 1.0,
    fields: Optional[Sequence[str]] = None,
    timesteps: Optional[Sequence[int]] = None,
    cmap: str = "RdBu_r",
    figsize: Tuple[float, float] = (2.4, 2.4),
) -> plt.Figure:
    C, L, _, _ = x.shape

    if timesteps is None:
        timesteps = list(range(L))

    if y is None:
        fig, axs = plt.subplots(
            nrows=L,
            ncols=C,
            figsize=(figsize[0] * C, figsize[1] * L),
            squeeze=False,
        )
    else:
        fig, axs = plt.subplots(
            nrows=3 * L,
            ncols=C,
            figsize=(figsize[0] * C, 3 * figsize[1] * L),
            squeeze=False,
        )

    for i in range(C):
        vmin = np.nanmin(x[i])
        vmax = np.nanmax(x[i])

        if fields:
            axs[0, i].set_title(f"{fields[i]}")

        for j in range(L):
            if y is None:
                axs[j, i].imshow(x[i, j], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none")
                axs[j, i].set_xticks([])
                axs[j, i].set_yticks([])

                axs[j, 0].set_ylabel(rf"$x_{{{timesteps[j]}}}$")
            else:
                axs[3 * j, i].imshow(
                    x[i, j], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none"
                )
                axs[3 * j, i].set_xticks([])
                axs[3 * j, i].set_yticks([])

                axs[3 * j + 1, i].imshow(
                    y[i, j], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none"
                )
                axs[3 * j + 1, i].set_xticks([])
                axs[3 * j + 1, i].set_yticks([])

                axs[3 * j + 2, i].imshow(
                    y[i, j] - x[i, j],
                    cmap="RdBu_r",
                    vmin=-tolerance,
                    vmax=tolerance,
                    interpolation="none",
                )
                axs[3 * j + 2, i].set_xticks([])
                axs[3 * j + 2, i].set_yticks([])

                axs[3 * j, 0].set_ylabel(rf"$x_{{{timesteps[j]}}}$")
                axs[3 * j + 1, 0].set_ylabel(rf"$y_{{{timesteps[j]}}}$")
                axs[3 * j + 2, 0].set_ylabel(rf"$y_{{{timesteps[j]}}} - x_{{{timesteps[j]}}}$")

    fig.align_labels()
    fig.tight_layout()

    return fig


def isotropic_power_spectrum(
    x: ArrayLike,
    edges: Optional[ArrayLike] = None,
    spatial: int = 2,
) -> Tuple[Tensor, Tensor]:
    r"""Computes the isotropic power spectrum of a field.

    Adapted from :func:`the_well.benchmark.metrics.spectral.power_spectrum`.

    Arguments:
        x: A field tensor, with shape :math:`(*, L_1, ..., L_N)`.
        edges: The frequency bin edges (in cycles per unit), with shape :math:`(L + 1)`.
        spatial: The number of spatial dimensions :math:`N`.

    Returns:
        The power spectrum and the frequency bin edges, with shape :math:`(*, L)` and
        :math:`(L + 1)`, respectively.
    """

    x = torch.as_tensor(x)

    # Frequency
    k = []

    for i in range(spatial):
        k_i = torch.fft.fftfreq(x.shape[i - spatial], dtype=x.dtype, device=x.device)
        k.append(k_i)

    k2 = map(torch.square, k)
    k2_iso = sum(torch.meshgrid(*k2, indexing="ij"))
    k_iso = torch.sqrt(k2_iso)

    if edges is None:
        bins = math.floor(math.sqrt(k_iso.ndim) * min(k_iso.shape) / 2)
        edges = torch.linspace(0, k_iso.max(), bins + 1, dtype=x.dtype, device=x.device)
    else:
        edges = torch.as_tensor(edges)
        bins = len(edges) - 1

    indices = torch.clip(torch.bucketize(k_iso.flatten(), edges), max=bins - 1)
    counts = torch.bincount(indices, minlength=bins)

    # Power spectrum
    s = torch.fft.fftn(x, dim=tuple(range(-spatial, 0)))
    p = torch.square(torch.abs(s))
    p = torch.flatten(p, start_dim=-spatial)

    p_iso = torch.zeros((*p.shape[:-1], bins), dtype=x.dtype, device=x.device)
    p_iso = p_iso.scatter_add(dim=-1, index=indices.expand_as(p), src=p)
    p_iso = p_iso / torch.clip(counts, min=1)

    return p_iso, edges


def draw_psd(
    x: ArrayLike,
    y: Optional[ArrayLike] = None,
    fields: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (3.2, 3.2),
) -> plt.Figure:
    C, *shape = x.shape

    fig, axs = plt.subplots(
        nrows=1,
        ncols=C,
        figsize=(figsize[0] * C, figsize[1]),
        squeeze=False,
    )

    for i in range(C):
        if fields:
            axs[0, i].set_title(f"{fields[i]}")

        x_ps, edges = isotropic_power_spectrum(x[i], spatial=len(shape))

        axs[0, i].loglog(edges[1:], x_ps, base=2, label="$x$", color="black")

        if y is not None:
            y_ps, _ = isotropic_power_spectrum(y[i], spatial=len(shape))

            axs[0, i].loglog(edges[1:], y_ps, base=2, label="$y$", color="C0", alpha=0.75)

        axs[0, i].set_xticks([2**i for i in range(math.floor(math.log2(edges[1].item())), 0)])

    axs[0, 0].set_ylabel("power spectrum density")
    axs[0, 0].legend()

    fig.align_labels()
    fig.tight_layout()

    return fig


def field2rgb(
    x: ArrayLike,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "magma",
) -> ArrayLike:
    x = np.asarray(x)

    if vmin is None:
        vmin = np.min(x)
    if vmax is None:
        vmax = np.max(x)

    palette = plt.get_cmap(cmap)

    x = (x - vmin) / (vmax - vmin)
    x = palette(x)
    x = 256 * x[..., :3]
    x = x.astype(np.uint8)

    return x


def draw_grid(
    x: ArrayLike,
    pad: int = 4,
    **kwargs,
) -> Image.Image:
    x = field2rgb(x, **kwargs)

    while x.ndim < 5:
        x = x[None]

    M, N, H, W, _ = x.shape

    img = Image.new(
        mode="RGB",
        size=(
            N * (W + pad) + pad,
            M * (H + pad) + pad,
        ),
        color=(255, 255, 255),
    )

    for i in range(M):
        for j in range(N):
            offset = (
                j * (W + pad) + pad,
                i * (H + pad) + pad,
            )

            img.paste(Image.fromarray(x[i][j]), offset)

    return img


def save_gif(
    x: ArrayLike,
    file: str,
    wait: float = 0.2,
    **kwargs,
):
    x = field2rgb(x, **kwargs)

    imgs = [Image.fromarray(img) for img in x]
    imgs[0].save(
        file,
        format="GIF",
        save_all=True,
        append_images=imgs[1:],
        duration=int(1000 * wait),
        loop=0,
    )
