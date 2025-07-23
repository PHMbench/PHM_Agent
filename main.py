from __future__ import annotations

"""Unified entry point for CLI and UI execution."""

import argparse
from dotenv import load_dotenv

from agents_config import create_manager_agent
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

    run_result = manager_agent.run(
        "If the US keeps it 2024 growth rate, how many years would it take for the GDP to double?"
    )
    print("Here is the token usage for the manager agent", run_result.token_usage)
    print("Here are the timing informations for the manager agent:", run_result.timing)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the PHM agent demo")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--inspect", action="store_true", help="Enable OpenInference instrumentation")
    parser.add_argument("--ui", action="store_true", help="Launch a Gradio web UI")
    args = parser.parse_args()
    main(args.config, args.inspect, args.ui)
