from __future__ import annotations

"""Utility for accessing local benchmark data."""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

__all__ = ["BenchmarkDataset"]


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
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run benchmark/generate_dummy_data.py to create it"
            )
        with h5py.File(path, "r") as f:
            data = f["data"][:]
        return data
