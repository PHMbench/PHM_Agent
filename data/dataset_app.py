"""Command line helper for selecting PHM dataset splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from agents_config import create_manager_agent
from model_config import configure_model
from .dataset_utils import load_metadata, get_signal, parse_id_list


def main() -> None:
    parser = argparse.ArgumentParser(description="PHM dataset helper")
    parser.add_argument("metadata", type=Path, help="Path to metadata Excel/CSV")
    parser.add_argument("data", type=Path, help="Path to HDF5 data file")
    parser.add_argument("--train", default="", help="Comma separated training IDs")
    parser.add_argument("--val", default="", help="Comma separated validation IDs")
    parser.add_argument("--test", default="", help="Comma separated test IDs")
    parser.add_argument("--model", default="litellm", help="Model backend for agent")
    args = parser.parse_args()

    df = load_metadata(args.metadata)
    print("Metadata:")
    print(df.head())

    train_ids = parse_id_list(args.train)
    val_ids = parse_id_list(args.val)
    test_ids = parse_id_list(args.test)

    model = configure_model(args.model)
    agent = create_manager_agent(model)

    for subset, ids in {"train": train_ids, "val": val_ids, "test": test_ids}.items():
        for sid in ids:
            _ = get_signal(args.data, sid)
            prompt = f"Process {subset} sample {sid}"
            print(agent.run(prompt))


if __name__ == "__main__":
    main()
