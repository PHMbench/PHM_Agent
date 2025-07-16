from __future__ import annotations

"""Launch a simple Gradio UI for the manager agent."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from agents_config import create_manager_agent
from model_config import configure_model
from smolagents import GradioUI
from dotenv import load_dotenv

load_dotenv(override=True)

def main() -> None:
    model = configure_model('litellm')
    agent = create_manager_agent(model)
    GradioUI(agent, file_upload_folder="./data").launch()


if __name__ == "__main__":
    main()
