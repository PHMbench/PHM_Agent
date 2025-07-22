"""Utility functions for loading and splitting PHM datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py
import pandas as pd
import numpy as np

__all__ = [
    "load_metadata",
    "load_hdf5",
    "get_signal",
    "parse_id_list",
]


def load_metadata(path: str | Path) -> pd.DataFrame:
    """Load metadata from an Excel or CSV file."""
    p = Path(path)
    if p.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(p)
    return pd.read_csv(p)


def load_hdf5(path: str | Path) -> h5py.File:
    """Open an HDF5 file in read-only mode."""
    return h5py.File(Path(path), "r")


def get_signal(h5_path: str | Path, sample_id: int) -> np.ndarray:
    """Return a dataset from ``h5_path`` indexed by ``sample_id``."""
    with h5py.File(h5_path, "r") as f:
        if str(sample_id) not in f:
            raise KeyError(f"Sample id {sample_id} not found")
        return f[str(sample_id)][:]


def parse_id_list(ids: str | Iterable[int]) -> list[int]:
    """Parse a comma separated string into a list of integers."""
    if isinstance(ids, str):
        if not ids:
            return []
        return [int(x.strip()) for x in ids.split(',') if x.strip()]
    return list(map(int, ids))
