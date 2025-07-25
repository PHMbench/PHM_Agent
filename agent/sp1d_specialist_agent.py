from __future__ import annotations

"""Specialist agent focused on 1-D signal processing."""

from typing import Any

from smolagents.agents import ToolCallingAgent
from smolagents.models import Model

from PHM_tools.Signal_processing import (
    normalize,
    detrend,
    fft,
    cepstrum,
    envelope_spectrum,
)
from utils.registry import register_agent


@register_agent("SP1D_SpecialistAgent")
class SP1DSpecialistAgent(ToolCallingAgent):
    """Agent wrapping common 1-D signal processing tools."""

    def __init__(self, model: Model, **kwargs: Any) -> None:
        tools = [normalize, detrend, fft, cepstrum, envelope_spectrum]
        instructions = "Apply 1-D signal processing tools as requested."
        super().__init__(
            tools=tools,
            model=model,
            name="sp1d_specialist",
            description="Expert in 1-D signal processing",
            instructions=instructions,
            return_full_result=True,
            add_base_tools=True,
            **kwargs,
        )


if __name__ == "__main__":
    class DummyModel:
        def __call__(self, messages):
            return type("Obj", (), {"content": "ok"})()

    agent = SP1DSpecialistAgent(DummyModel())
    print("Available tools:", [t.name for t in agent.tools])
