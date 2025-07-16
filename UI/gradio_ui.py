"""Gradio UI demo that lets the agent plan tool usage."""

from __future__ import annotations

from smolagents import CodeAgent, GradioUI

from agents_config import create_manager_agent
from model_config import configure_model


def launch_ui(inference: str = "inference_client") -> None:
    """Launch a web UI for the manager agent."""
    model = configure_model(inference)
    agent = create_manager_agent(model)
    GradioUI(agent, file_upload_folder="./uploads").launch()


if __name__ == "__main__":
    launch_ui()
