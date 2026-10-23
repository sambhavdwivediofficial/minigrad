from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence
import numpy as np


class Dataset:

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError


class TensorDataset(Dataset):

    def __init__(self, *arrays: np.ndarray):
        if not arrays:
            raise ValueError("TensorDataset requires at least one array")
        self.arrays = [np.asarray(a, dtype=np.float64) for a in arrays]
        lengths = {a.shape[0] for a in self.arrays}
        if len(lengths) != 1:
            raise ValueError(f"all arrays must share the same first dimension, got sizes {lengths}")

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, index):
        return tuple(a[index] for a in self.arrays)


class CSVDataset(Dataset):

    def __init__(
        self,
        path: str | Path,
        target_column: int = -1,
        feature_columns: Sequence[int] | None = None,
        has_header: bool = True,
        delimiter: str = ",",
    ):
        raw = np.genfromtxt(
            Path(path), delimiter=delimiter, skip_header=1 if has_header else 0, dtype=np.float64
        )
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if np.isnan(raw).any():
            raise ValueError(
                "CSVDataset found non-numeric or missing values; this loader supports "
                "clean numeric tables only (see class docstring)."
            )
        num_cols = raw.shape[1]
        target_idx = target_column if target_column >= 0 else num_cols + target_column
        if feature_columns is None:
            feature_columns = [c for c in range(num_cols) if c != target_idx]
        self.x = raw[:, list(feature_columns)]
        self.y = raw[:, [target_idx]]

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class DataLoader:

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = False,
        seed: int | None = None,
        drop_last: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[tuple]:
        n = len(self.dataset)
        indices = np.arange(n)
        if self.shuffle:
            self._rng.shuffle(indices)
        for start in range(0, n, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            if self.drop_last and len(batch_idx) < self.batch_size:
                continue
            if isinstance(self.dataset, TensorDataset):
                yield tuple(a[batch_idx] for a in self.dataset.arrays)
            else:
                items = [self.dataset[i] for i in batch_idx]
                num_fields = len(items[0])
                yield tuple(np.stack([item[f] for item in items]) for f in range(num_fields))

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return int(np.ceil(n / self.batch_size))
