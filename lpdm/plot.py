r"""Plotting and image helpers."""

import matplotlib.pyplot as plt
import numpy as np

from numpy.typing import ArrayLike
from PIL import Image
from typing import Optional, Sequence, Tuple


def draw_fields(
    x: ArrayLike,
    y: Optional[ArrayLike] = None,
    tolerance: float = 1.0,
    fields: Optional[Sequence[str]] = None,
    timesteps: Optional[Sequence[int]] = None,
    cmap: str = "RdBu_r",
    figsize: Tuple[float, float] = (2.4, 2.4),
) -> plt.Figure:
    L, _, _, C = x.shape

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
        vmin = np.nanmin(x[..., i])
        vmax = np.nanmax(x[..., i])

        if fields:
            axs[0, i].set_title(f"{fields[i]}")

        for j in range(L):
            if y is None:
                axs[j, i].imshow(
                    x[j, ..., i], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none"
                )
                axs[j, i].set_xticks([])
                axs[j, i].set_yticks([])

                axs[j, 0].set_ylabel(rf"$x_{{{timesteps[j]}}}$")
            else:
                axs[3 * j, i].imshow(
                    x[j, ..., i], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none"
                )
                axs[3 * j, i].set_xticks([])
                axs[3 * j, i].set_yticks([])

                axs[3 * j + 1, i].imshow(
                    y[j, ..., i], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none"
                )
                axs[3 * j + 1, i].set_xticks([])
                axs[3 * j + 1, i].set_yticks([])

                axs[3 * j + 2, i].imshow(
                    y[j, ..., i] - x[j, ..., i],
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
    fig.set_layout_engine(layout="tight")

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
    dt: float = 0.2,
    **kwargs,
):
    x = field2rgb(x, **kwargs)

    imgs = [Image.fromarray(img) for img in x]
    imgs[0].save(
        file,
        format="GIF",
        save_all=True,
        append_images=imgs[1:],
        duration=int(1000 * dt),
        loop=0,
    )
