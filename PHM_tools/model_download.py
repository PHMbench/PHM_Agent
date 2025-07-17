from __future__ import annotations

"""Utility for retrieving popular models from the Hugging Face Hub."""

from huggingface_hub import list_models
from smolagents import tool

from utils.registry import register_tool


@register_tool("model_download_tool")
@tool
def model_download_tool(task: str) -> str:
    """Return the most downloaded model for a given pipeline task on the Hub.

    Args:
        task: Name of the pipeline tag, for example ``text-classification``.

    Returns:
        Repository ID of the model with the highest download count.
    """
    models = list_models(
        filter={"pipeline_tag": task}, sort="downloads", direction=-1, limit=1
    )
    if not models:
        raise ValueError(f"No models found for task '{task}'")
    return models[0].modelId
