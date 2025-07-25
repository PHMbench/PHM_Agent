from __future__ import annotations

"""Signal slicing utilities for PHM workflows."""

from typing import Any, Dict

from PHM_tools.Signal_processing import normalize
from PHM_tools.data_management import DataManager


def apply_transform(
    data_manager: DataManager,
    source_data_key: str,
    transform_tool_name: str,
    transform_params: Dict[str, Any],
) -> str:
    """Apply a transformation to stored data and save the result.

    Args:
        data_manager: Central data manager instance.
        source_data_key: Key of the source data in :class:`DataManager`.
        transform_tool_name: Name of the tool to apply (e.g., ``'normalize'``).
        transform_params: Parameters passed to the tool.

    Returns:
        Key under which the transformed data is stored.
    """
    data = data_manager.get_processed_data(source_data_key)
    if data is None:
        raise KeyError(f"Data key '{source_data_key}' not found")

    # In a real system we'd lookup the tool dynamically. Here we demo with 'normalize'.
    if transform_tool_name == "normalize":
        transformed = normalize(data)
    else:
        raise ValueError(f"Unsupported transform '{transform_tool_name}'")

    new_key = f"{source_data_key}_{transform_tool_name}"
    data_manager.store_processed_data(new_key, transformed)
    return new_key


def extract_slice(
    data_manager: DataManager, source_data_key: str, slice_params: Dict[str, Any]
) -> str:
    """Extract a slice from stored data using ``slice`` semantics.

    Args:
        data_manager: Central data manager instance.
        source_data_key: Key of the data to slice.
        slice_params: Slice specification, e.g., ``{"start": 0, "stop": 10}``.

    Returns:
        Key under which the slice is stored.
    """
    data = data_manager.get_processed_data(source_data_key)
    if data is None:
        raise KeyError(f"Data key '{source_data_key}' not found")

    start = slice_params.get("start")
    stop = slice_params.get("stop")
    sliced = data[start:stop]

    new_key = f"{source_data_key}_slice_{start}_{stop}"
    data_manager.store_processed_data(new_key, sliced)
    return new_key


if __name__ == "__main__":
    import numpy as np
    import tempfile
    import pandas as pd
    import h5py

    rng = np.random.default_rng(0)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp_csv:
        pd.DataFrame({"id": ["x"], "label": ["demo"]}).to_csv(tmp_csv.name, index=False)
        csv_path = tmp_csv.name
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_h5:
        with h5py.File(tmp_h5.name, "w") as f:
            f.create_dataset("x", data=rng.normal(size=20))
        h5_path = tmp_h5.name

    dm = DataManager(csv_path, h5_path)
    dm.store_processed_data("raw", rng.normal(size=20))
    key_norm = apply_transform(dm, "raw", "normalize", {})
    key_slice = extract_slice(dm, key_norm, {"start": 0, "stop": 5})
    print("Slice key:", key_slice, "data:", dm.get_processed_data(key_slice))
