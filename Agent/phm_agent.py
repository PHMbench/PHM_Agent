from __future__ import annotations

"""Example PHM agent using the smolagents framework."""

from typing import Any

from smolagents.agents import ToolCallingAgent
from smolagents.models import Model
from smolagents.tools import Tool

from utils.registry import register_agent


@register_agent("PHMAgent")
class PHMAgent(ToolCallingAgent):
    """Minimal agent registered for later retrieval."""

    def __init__(self, tools: list[Tool], model: Model, **kwargs: Any) -> None:
        super().__init__(tools=tools, model=model, **kwargs)

