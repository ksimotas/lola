import glob, os
from pathlib import Path
from typing import Sequence, Union, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

TensorLike = Union[torch.Tensor, np.ndarray]

class LatentVolumeFolder(Dataset):
    """
    Loads 3D latent volumes from a directory tree.

    Expects files ending with .pt/.pth/.npy that contain either:
      - a dict with key `latent_key` -> array/tensor shaped [C,D,H,W] or [1,C,D,H,W], or
      - a bare array/tensor of that shape.

    Returns: {"image": x} with x: float32 tensor [C,D,H,W].
    Optional normalization: per-channel or scalar (x - mean) / std.
    """

    def __init__(
        self,
        root: Union[str, Path],
        exts: Sequence[str] = (".pt", ".pth", ".npy"),
        latent_key: Optional[str] = "z_q",
        item_index: int = 0,            # for list/tuple containers
        channels: int = 512,
        normalize_mean: Optional[Union[float, Sequence[float], TensorLike]] = None,
        normalize_std: Optional[Union[float, Sequence[float], TensorLike]] = None,
        max_samples: Optional[int] = None,
    ):
        self.root = str(root)
        self.paths = sorted(
            p for p in glob.glob(os.path.join(self.root, "**", "*.*"), recursive=True)
            if any(p.endswith(e) for e in exts)
        )
        if max_samples is not None:
            self.paths = self.paths[:max_samples]
        if not self.paths:
            raise FileNotFoundError(f"No latent files under {root} with exts={exts}")

        self.latent_key = latent_key
        self.item_index = item_index
        self.channels = channels

        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def _to_tensor(x: TensorLike) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        raise TypeError(f"Unsupported type: {type(x)}")

    @staticmethod
    def _unwrap(obj, item_index: int):
        # unwrap list/tuple containers and 0-d object arrays
        if isinstance(obj, (list, tuple)) and len(obj) > 0:
            obj = obj[item_index]
        if isinstance(obj, np.ndarray) and obj.dtype == object and obj.size == 1:
            obj = obj.item()
        return obj

    def _load_obj(self, p: str):
        if p.endswith((".pt", ".pth")):
            return torch.load(p, map_location="cpu")
        return np.load(p, allow_pickle=True)

    def _apply_norm(self, x: torch.Tensor) -> torch.Tensor:
        # x: [C,D,H,W], float32
        if self.normalize_mean is not None:
            if isinstance(self.normalize_mean, (list, tuple, np.ndarray, torch.Tensor)):
                m = torch.as_tensor(self.normalize_mean, dtype=x.dtype).view(-1,1,1,1)
                x = x - m
            else:
                x = x - float(self.normalize_mean)
        if self.normalize_std is not None:
            eps = 1e-6
            if isinstance(self.normalize_std, (list, tuple, np.ndarray, torch.Tensor)):
                s = torch.as_tensor(self.normalize_std, dtype=x.dtype).view(-1,1,1,1).clamp_min(eps)
                x = x / s
            else:
                x = x / max(float(self.normalize_std), eps)
        return x

    def __getitem__(self, i: int):
        p = self.paths[i]
        obj = self._unwrap(self._load_obj(p), self.item_index)

        # extract latent
        if isinstance(obj, dict):
            if self.latent_key is None or self.latent_key not in obj:
                keys = list(obj.keys())
                raise KeyError(f"{p}: missing key '{self.latent_key}', found keys={keys}")
            x = obj[self.latent_key]
            # NEW: unwrap if the value is a list/tuple/object-array (e.g., z_q[0])
            if isinstance(x, (list, tuple)) and len(x) > self.item_index:
                x = x[self.item_index]
            if isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
                x = x.item()
            x = self._to_tensor(x)
        else:
            x = self._to_tensor(obj)

        # shapes: [1,C,D,H,W] -> [C,D,H,W], or move last channel to first
        if x.ndim == 5 and x.shape[0] == 1:
            x = x[0]
        if x.ndim == 4 and x.shape[-1] == self.channels and x.shape[0] != self.channels:
            x = x.movedim(-1, 0)

        if x.ndim != 4:
            raise ValueError(f"{p}: expected 4D [C,D,H,W], got {tuple(x.shape)}")

        x = x.float().contiguous()
        x = self._apply_norm(x)
        return {"image": x}
