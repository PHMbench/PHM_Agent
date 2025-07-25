from __future__ import annotations

"""Central data manager for PHM tools.

This module defines :class:`DataManager`, a utility responsible for loading raw
signal data and metadata and storing intermediate processing results.
"""

from typing import Any, Dict, List

import h5py
import numpy as np
import pandas as pd


class DataManager:
    """Manage signal metadata and processed data."""

    def __init__(self, metadata_path: str, signal_data_path: str) -> None:
        self.metadata_df = pd.read_csv(metadata_path).set_index("id")
        self.signal_data_path = signal_data_path
        self.processed_data_cache: Dict[str, Any] = {}

    def get_all_ids(self, label: str | None = None) -> List[str]:
        df = self.metadata_df
        if label is not None:
            df = df[df["label"] == label]
        return df.index.astype(str).tolist()

    def get_signal_by_id(self, signal_id: str) -> Dict[str, np.ndarray]:
        with h5py.File(self.signal_data_path, "r") as f:
            if signal_id not in f:
                raise KeyError(f"Signal ID '{signal_id}' not found")
            data = f[signal_id][:]
        return {signal_id: data}

    def get_metadata_by_id(self, signal_id: str) -> Dict[str, Any]:
        return self.metadata_df.loc[signal_id].to_dict()

    def store_processed_data(self, key: str, data: Any) -> None:
        self.processed_data_cache[key] = data

    def get_processed_data(self, key: str) -> Any:
        return self.processed_data_cache.get(key)


if __name__ == "__main__":
    import tempfile

    rng = np.random.default_rng(0)
    # create temporary metadata CSV
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp_csv:
        pd.DataFrame({"id": ["a", "b"], "label": ["normal", "fault"]}).to_csv(tmp_csv.name, index=False)
        metadata_path = tmp_csv.name

    # create temporary HDF5 with random signals
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_h5:
        with h5py.File(tmp_h5.name, "w") as f:
            f.create_dataset("a", data=rng.normal(size=100))
            f.create_dataset("b", data=rng.normal(size=100))
        data_path = tmp_h5.name

    dm = DataManager(metadata_path, data_path)
    ids = dm.get_all_ids()
    sig = dm.get_signal_by_id(ids[0])
    print("IDs:", ids)
    print("Signal sample shape:", sig[ids[0]].shape)
