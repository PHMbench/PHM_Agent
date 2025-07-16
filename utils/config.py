import argparse
import yaml


MODEL_PROVIDER_MAP = {
    "gpt-4o": "openai",
    "gpt-3.5-turbo": "openai",
    "gemini-1.5-flash": "gemini",
    "gemini-pro": "gemini",
    'gemini-2.5-pro': 'gemini',
    "gemini-2.5-flash": "gemini",
}


def load_config(path):
    """Load a YAML configuration file into an argparse.Namespace"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return argparse.Namespace(**data)


def get_provider(model_name: str) -> str:
    """Map a model name to its API provider."""
    return MODEL_PROVIDER_MAP.get(model_name, model_name)
