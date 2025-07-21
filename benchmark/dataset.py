from __future__ import annotations

"""Utility for accessing local benchmark data."""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

__all__ = ["BenchmarkDataset", "generate_sample_files"]


@dataclass
class BenchmarkDataset:
    """Dataset wrapper for metadata-driven access to HDF5 files."""

    metadata_path: Path
    """Path to the CSV file describing sample locations."""

    def __post_init__(self) -> None:
        self.metadata_path = Path(self.metadata_path)
        self._metadata = pd.read_csv(self.metadata_path)

    @property
    def metadata(self) -> pd.DataFrame:
        """Loaded metadata table."""
        return self._metadata.copy()

    def load(self, sample_id: int) -> np.ndarray:
        """Load a data array given its ``sample_id``.

        Args:
            sample_id: Value from the ``id`` column of the metadata.

        Returns:
            Stored array for the selected sample.
        """
        row = self._metadata[self._metadata["id"] == sample_id]
        if row.empty:
            raise KeyError(f"Sample id {sample_id} not found")
        path = Path(row.iloc[0]["path"])
        with h5py.File(path, "r") as f:
            data = f["data"][:]
        return data


def generate_sample_files(metadata_path: str, seed: int = 0) -> None:
    """Generate example HDF5 files referenced by ``metadata_path``.

    This utility creates small random datasets to match the CSV metadata. It is
    useful for demos and tests where the binary files are not included in the
    repository.

    Args:
        metadata_path: CSV file describing sample locations and parameters.
        seed: Random seed for reproducibility.
    """

    meta = pd.read_csv(metadata_path)
    rng = np.random.default_rng(seed)
    for _, row in meta.iterrows():
        out = Path(row["path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        data = rng.normal(scale=row.get("noise_std", 0.1), size=(100, 4))
        with h5py.File(out, "w") as f:
            f.create_dataset("data", data=data)
