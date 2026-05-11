from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import numpy as np
try:
    import torch
    from torch.utils import data
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal test environments
    torch = None

    class _DatasetBase:
        pass

    class _DataNamespace:
        Dataset = _DatasetBase

    data = _DataNamespace()


@dataclass(frozen=True)
class CacheInfo:
    source_path: Path
    cache_path: Path
    shape: tuple[int, int]
    dtype: np.dtype


def default_cache_path(source_path, cache_dir):
    source_path = Path(source_path)
    cache_dir = Path(cache_dir)
    return cache_dir / f"{source_path.stem}.npy"


def cache_text_dataset(source_path, cache_path, dtype=np.float32) -> CacheInfo:
    source_path = Path(source_path)
    cache_path = Path(cache_path)
    if cache_path.exists():
        cached = np.load(cache_path, mmap_mode="r")
        return CacheInfo(
            source_path=source_path,
            cache_path=cache_path,
            shape=tuple(int(dim) for dim in cached.shape),
            dtype=np.dtype(cached.dtype),
        )

    matrix = np.loadtxt(source_path, dtype=dtype)
    np.save(cache_path, matrix.astype(dtype, copy=False))
    return CacheInfo(
        source_path=source_path,
        cache_path=cache_path,
        shape=tuple(int(dim) for dim in matrix.shape),
        dtype=np.dtype(dtype),
    )


def ensure_cached_dataset(source_path, cache_dir, dtype=np.float32) -> CacheInfo:
    cache_path = default_cache_path(source_path, cache_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    return cache_text_dataset(source_path, cache_path, dtype=dtype)


def open_memmap(cache_path):
    return np.load(Path(cache_path), mmap_mode="r")


class CachedArrayView:
    def __init__(self, array: np.ndarray, start_col: int = 0, end_col: int | None = None):
        self.array = array
        self.start_col = start_col
        self.end_col = array.shape[1] if end_col is None else end_col

    @property
    def shape(self) -> tuple[int, int]:
        return (self.array.shape[0], self.end_col - self.start_col)

    def columns(self, start: int, stop: int) -> np.ndarray:
        return self.array[:, self.start_col + start : self.start_col + stop]


class ConcatenatedCachedArrayView:
    def __init__(self, views):
        if not views:
            raise ValueError("views must not be empty")
        row_count = views[0].shape[0]
        for view in views:
            if view.shape[0] != row_count:
                raise ValueError("all views must have the same number of rows")
        self.views = list(views)
        self._offsets = []
        total = 0
        for view in self.views:
            self._offsets.append(total)
            total += view.shape[1]
        self._shape = (row_count, total)

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    def columns(self, start: int, stop: int) -> np.ndarray:
        pieces = []
        for view, offset in zip(self.views, self._offsets):
            view_start = max(start - offset, 0)
            view_stop = min(stop - offset, view.shape[1])
            if view_start < view_stop:
                pieces.append(view.columns(view_start, view_stop))
        if not pieces:
            return self.views[0].columns(0, 0)
        if len(pieces) == 1:
            return pieces[0]
        return np.concatenate(pieces, axis=1)


def split_train_validation(array: np.ndarray, train_fraction: float = 0.8):
    split_idx = int(np.floor(array.shape[1] * train_fraction))
    return CachedArrayView(array, end_col=split_idx), CachedArrayView(array, start_col=split_idx)


def recommended_num_workers(max_workers: int = 4) -> int:
    cpu_count = os.cpu_count() or 1
    return max(0, min(max_workers, cpu_count - 1))


class SlidingWindowDataset(data.Dataset):
    def __init__(self, view: CachedArrayView, k: int, num_classes: int, T: int):
        self.view = view
        self.k = k
        self.num_classes = num_classes
        self.T = T
        self.length = self.view.shape[1] - self.T + 1

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        stop = index + self.T
        window = self.view.columns(index, stop)
        x = np.array(window[:40, :].T, dtype=np.float32, copy=True)
        y_source = self.view.columns(stop - 1, stop)
        y = np.int64(y_source[-5 + self.k, 0] - 1)
        if torch is None:
            return x[None, :, :], np.array(y, dtype=np.int64)
        return torch.from_numpy(x[None, :, :]), torch.tensor(y, dtype=torch.int64)
