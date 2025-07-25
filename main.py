from __future__ import annotations

"""Unified entry point for CLI and UI execution."""

import argparse
from dotenv import load_dotenv

from agents_config import create_manager_agent
import numpy as np
import pandas as pd
import h5py
import tempfile
from model_config import configure_model
from utils.config import load_config
from UI.enhanced_ui import EnhancedGradioUI

load_dotenv(override=True)


def main(config: str = "config.yaml", inspect: bool = False, ui: bool = False) -> None:
    """Run the PHM agent demo or launch the UI.

    Parameters
    ----------
    config:
        Path to the YAML configuration file.
    inspect:
        Enable OpenInference instrumentation when ``True``.
    ui:
        Launch the Gradio interface instead of running a demo query.
    """
    cfg = load_config(config)
    if inspect:
        from openinference.instrumentation.smolagents import SmolagentsInstrumentor
        from phoenix.otel import register

        register()
        SmolagentsInstrumentor().instrument(skip_dep_check=True)

    model = configure_model(cfg.inference, cfg.model_id)
    manager_agent = create_manager_agent(model, cfg)

    if ui:
        EnhancedGradioUI(
            manager_agent,
            file_upload_folder="./uploads",
            config=cfg,
        ).launch()
        return

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp_csv:
        pd.DataFrame({"id": ["sample"], "label": ["normal"]}).to_csv(tmp_csv.name, index=False)
        metadata_path = tmp_csv.name

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_h5:
        with h5py.File(tmp_h5.name, "w") as f:
            f.create_dataset("sample", data=np.random.randn(500))
        signal_path = tmp_h5.name

    payload = {
        "metadata_path": metadata_path,
        "signal_path": signal_path,
        "reference_ids": ["sample"],
        "test_id": "sample",
    }

    result = manager_agent.run_workflow(payload)
    print("Workflow result:", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the PHM agent demo")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--inspect", action="store_true", help="Enable OpenInference instrumentation")
    parser.add_argument("--ui", action="store_true", help="Launch a Gradio web UI")
    args = parser.parse_args()
    main(args.config, args.inspect, args.ui)
