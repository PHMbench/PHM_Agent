from __future__ import annotations

"""Agent for analysing physical fault signatures."""

from typing import Any

from smolagents.agents import ToolCallingAgent
from smolagents.models import Model

from PHM_tools.Signal_processing import envelope_spectrum
from PHM_tools.Feature_extracting import extract_time_features
from utils.registry import register_agent


@register_agent("PhysicalFeatureAgent")
class PhysicalFeatureAgent(ToolCallingAgent):
    """Agent specialised in physics-based feature analysis."""

    def __init__(self, model: Model, **kwargs: Any) -> None:
        tools = [envelope_spectrum, extract_time_features]
        instructions = "Analyse physical fault signatures using available tools."
        super().__init__(
            tools=tools,
            model=model,
            name="physical_feature_agent",
            description="Analyses physics-based features",
            instructions=instructions,
            return_full_result=True,
            add_base_tools=True,
            **kwargs,
        )


if __name__ == "__main__":
    class DummyModel:
        def __call__(self, messages):
            return type("Obj", (), {"content": "ok"})()

    agent = PhysicalFeatureAgent(DummyModel())
    print("Physical tools:", [t.name for t in agent.tools])
