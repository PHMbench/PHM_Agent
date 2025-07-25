from __future__ import annotations

"""Agent dedicated to extracting statistical features."""

from typing import Any

from smolagents.agents import ToolCallingAgent
from smolagents.models import Model

from PHM_tools.Feature_extracting import (
    extract_time_features,
    extract_frequency_features,
)
from utils.registry import register_agent


@register_agent("StatsFeatureAgent")
class StatsFeatureAgent(ToolCallingAgent):
    """Agent that extracts time and frequency features."""

    def __init__(self, model: Model, **kwargs: Any) -> None:
        tools = [extract_time_features, extract_frequency_features]
        instructions = "Extract statistical features from signals."
        super().__init__(
            tools=tools,
            model=model,
            name="stats_feature_agent",
            description="Computes statistical features",
            instructions=instructions,
            return_full_result=True,
            add_base_tools=True,
            **kwargs,
        )


if __name__ == "__main__":
    class DummyModel:
        def __call__(self, messages):
            return type("Obj", (), {"content": "ok"})()

    agent = StatsFeatureAgent(DummyModel())
    print("Feature tools:", [t.name for t in agent.tools])
