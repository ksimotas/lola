r"""Plotting and image helpers."""

import math
import matplotlib.animation as ani
import matplotlib.pyplot as plt
import numpy as np
import torch

from numpy.typing import ArrayLike
from PIL import Image
from typing import Optional, Sequence, Tuple

from .fourier import isotropic_power_spectrum


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

    if torch.is_tensor(x):
        x = x.numpy(force=True)

    if torch.is_tensor(y):
        y = y.numpy(force=True)

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

    if torch.is_tensor(x):
        x = x.numpy(force=True)

    if torch.is_tensor(y):
        y = y.numpy(force=True)

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


def draw_psd(
    x: ArrayLike,
    y: Optional[ArrayLike] = None,
    fields: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (3.2, 3.2),
) -> plt.Figure:
    C, *shape = x.shape

    if torch.is_tensor(x):
        x = x.numpy(force=True)

    if torch.is_tensor(y):
        y = y.numpy(force=True)

    fig, axs = plt.subplots(
        nrows=1,
        ncols=C,
        figsize=(figsize[0] * C, figsize[1]),
        squeeze=False,
    )

    for i in range(C):
        if fields:
            axs[0, i].set_title(f"{fields[i]}")

        p_x, k = isotropic_power_spectrum(x[i], spatial=len(shape))

        axs[0, i].loglog(1 / k, p_x, base=2, label="$x$", color="black")

        if y is not None:
            p_y, _ = isotropic_power_spectrum(y[i], spatial=len(shape))

            axs[0, i].loglog(1 / k, p_y, base=2, label="$y$", color="C0", alpha=0.75)

        axs[0, i].invert_xaxis()
        axs[0, i].set_xticks([2**i for i in range(1, math.ceil(math.log2(1 / k[0].item())))])

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
    if torch.is_tensor(x):
        x = x.numpy(force=True)

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
