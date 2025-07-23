from __future__ import annotations

"""Configuration utilities for the PHM Agent demo."""

from dataclasses import dataclass, field
from typing import List
import yaml

MODEL_PROVIDER_MAP = {
    "gpt-4o": "openai",
    "gpt-3.5-turbo": "openai",
    "gemini-1.5-flash": "gemini",
    "gemini-pro": "gemini",
    "gemini-2.5-pro": "gemini",
    "gemini-2.5-flash": "gemini",
}


@dataclass
class Config:
    """Application configuration loaded from ``config.yaml``."""

    inference: str = "litellm"
    model_id: str = "gemini/gemini-2.5-pro"
    enabled_agents: List[str] = field(
        default_factory=lambda: ["search_agent", "phm_agent", "retrieval_agent"]
    )


def load_config(path: str) -> Config:
    """Return configuration parsed from ``path``."""

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if "enabled_agents" not in data:
        data["enabled_agents"] = [
            "search_agent",
            "phm_agent",
            "retrieval_agent",
        ]

    return Config(**data)


def get_provider(model_name: str) -> str:
    """Map a model name to its API provider."""

    return MODEL_PROVIDER_MAP.get(model_name, model_name)
