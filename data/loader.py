from __future__ import annotations

"""Dataset utilities for PHM analysis.

This module centralizes data loading functions. It reads metadata from Excel
files using ``pandas`` and provides helpers to retrieve signal arrays stored in
HDF5 format.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd

__all__ = ["PHMDataset"]


@dataclass
class PHMDataset:
    """Simple dataset wrapper combining metadata and HDF5 signals."""

    metadata_path: Path
    data_path: Path

    def __post_init__(self) -> None:
        self.metadata_path = Path(self.metadata_path)
        self.data_path = Path(self.data_path)
        self._metadata = self._load_metadata()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_metadata(self) -> pd.DataFrame:
        """Load the Excel metadata file."""
        if not self.metadata_path.exists():
            raise FileNotFoundError(self.metadata_path)
        ext = self.metadata_path.suffix.lower()
        engine = "openpyxl" if ext in {".xlsx", ".xlsm"} else "xlrd"
        return pd.read_excel(self.metadata_path, engine=engine)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def metadata(self) -> pd.DataFrame:
        """Return a copy of the loaded metadata table."""
        return self._metadata.copy()

    def load(self, sample_id: int) -> np.ndarray:
        """Load a signal by ``sample_id``.

        Parameters
        ----------
        sample_id:
            Value present in the ``id`` column of the metadata.

        Returns
        -------
        numpy.ndarray
            Stored signal array associated with ``sample_id``.
        """
        with h5py.File(self.data_path, "r") as f:
            if str(sample_id) not in f:
                raise KeyError(f"ID {sample_id} not found in {self.data_path}")
            return f[str(sample_id)][:]
