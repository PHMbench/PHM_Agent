from __future__ import annotations

"""Utility program to preview metadata and define data splits."""

import argparse
import json
from pathlib import Path

import gradio as gr

from agents_config import create_manager_agent
from model_config import configure_model
from .loader import PHMDataset


def run_agent(prompt: str, inference: str = "litellm") -> str:
    """Invoke the PHM agent with ``prompt`` and return the response."""
    model = configure_model(inference)
    agent = create_manager_agent(model)
    result = agent.run(prompt)
    return str(result.final_answer)


def launch_ui(metadata_path: str, data_path: str):
    dataset = PHMDataset(metadata_path, data_path)

    def submit(train_ids, val_ids, test_ids):
        prompt = (
            "Use the provided dataset to perform PHM analysis.\n"
            f"Training IDs: {train_ids}\n"
            f"Validation IDs: {val_ids}\n"
            f"Test IDs: {test_ids}"
        )
        return run_agent(prompt)

    with gr.Blocks(title="Dataset Splitter") as demo:
        gr.Markdown("## Metadata preview")
        gr.DataFrame(dataset.metadata)
        train = gr.Textbox(label="Train IDs (comma separated)")
        val = gr.Textbox(label="Validation IDs")
        test = gr.Textbox(label="Test IDs")
        out = gr.Textbox(label="Agent Output")
        btn = gr.Button("Run Agent")
        btn.click(submit, [train, val, test], out)
    demo.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset splits")
    parser.add_argument("metadata", help="Path to Excel metadata")
    parser.add_argument("data", help="Path to HDF5 data")
    args = parser.parse_args()
    launch_ui(args.metadata, args.data)
