from __future__ import annotations

"""Specialist agent for 2-D signal representations."""

from typing import Any

from smolagents.agents import ToolCallingAgent
from smolagents.models import Model

from PHM_tools.Signal_processing import (
    spectrogram,
    mel_spectrogram,
    scalogram,
)
from utils.registry import register_agent


@register_agent("SP2D_SpecialistAgent")
class SP2DSpecialistAgent(ToolCallingAgent):
    """Agent providing 2-D signal processing capabilities."""

    def __init__(self, model: Model, **kwargs: Any) -> None:
        tools = [spectrogram, mel_spectrogram, scalogram]
        instructions = "Produce 2-D signal representations as needed."
        super().__init__(
            tools=tools,
            model=model,
            name="sp2d_specialist",
            description="Expert in 2-D signal processing",
            instructions=instructions,
            return_full_result=True,
            add_base_tools=True,
            **kwargs,
        )


if __name__ == "__main__":
    class DummyModel:
        def __call__(self, messages):
            return type("Obj", (), {"content": "ok"})()

    agent = SP2DSpecialistAgent(DummyModel())
    print("Tools:", [t.name for t in agent.tools])
