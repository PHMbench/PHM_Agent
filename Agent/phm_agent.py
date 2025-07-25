from __future__ import annotations

"""Predictive Health Management analysis agent.

This agent acts as a professional PHM diagnostician. It dynamically
selects from the registered PHM tools to process uploaded data, perform
feature extraction and anomaly detection, and summarise the findings.
"""

from typing import Any

from smolagents.agents import ToolCallingAgent
from smolagents.models import Model
from smolagents.tools import Tool

from utils.registry import register_agent


@register_agent("PHMAgent")
class PHMAgent(ToolCallingAgent):
    """Minimal agent registered for later retrieval."""

    def __init__(self, tools: list[Tool], model: Model, **kwargs: Any) -> None:
        instructions = (
            "You are a predictive maintenance diagnostics expert. "
            "Analyse sensor signals using the available PHM tools, "
            "detect anomalies or faults and provide clear reasoning "
            "behind your conclusions."
        )
        super().__init__(
            tools=tools,
            model=model,
            name="phm_agent",
            description="Expert agent for PHM signal analysis and diagnosis.",
            instructions=instructions,
            **kwargs,
        )


if __name__ == "__main__":
    class DummyModel:
        def __call__(self, messages):
            return type("Obj", (), {"content": "ok"})()

    agent = PHMAgent([], DummyModel())
    print("Created PHMAgent with", len(agent.tools), "tools")

