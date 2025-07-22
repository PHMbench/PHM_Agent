from __future__ import annotations

"""Utility for accessing local benchmark data.

The :class:`BenchmarkDataset` class provides a simple interface for loading
example vibration signals stored as HDF5 files. A small helper is included to
generate synthetic data so the examples can run without downloading anything.
"""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

__all__ = ["BenchmarkDataset", "create_example_dataset"]


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


def create_example_dataset(output_dir: str, num_samples: int = 2) -> Path:
    """Generate a tiny synthetic dataset for demonstration purposes.

    Each sample is saved as an HDF5 file referenced by a metadata CSV. The
    generated files mimic four-channel vibration recordings with additive noise.

    Args:
        output_dir: Destination directory for the data files and metadata.
        num_samples: Number of samples to create.

    Returns:
        Path to the generated metadata CSV file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx in range(num_samples):
        signal = np.random.normal(scale=0.05, size=(4, 1024))
        file_path = out / f"{idx}.h5"
        with h5py.File(file_path, "w") as f:
            f.create_dataset("data", data=signal)
        rows.append({
            "id": idx,
            "system": 0,
            "domain": 0,
            "excitation": 0,
            "repeat": idx,
            "fs": 1000,
            "noise_std": 0.05,
            "path": str(file_path)
        })

    df = pd.DataFrame(rows)
    metadata_path = out / "metadata.csv"
    df.to_csv(metadata_path, index=False)
    return metadata_path
