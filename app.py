from __future__ import annotations

"""Entry point for launching the Gradio UI."""

from UI.enhanced_ui import EnhancedGradioUI
from agents_config import create_manager_agent
from model_config import configure_model
from utils.config import load_config


def main(config_path: str = "config.yaml") -> None:
    """Launch the Gradio interface with the configured model."""
    cfg = load_config(config_path)
    model = configure_model(cfg.inference, cfg.model_id)
    agent = create_manager_agent(model, cfg)
    ui = EnhancedGradioUI(agent, file_upload_folder="./uploads", config=cfg)
    ui.launch()


if __name__ == "__main__":
    main()
