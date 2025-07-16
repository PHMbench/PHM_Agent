"""Simple registry utilities for PHM tools and agents."""

from __future__ import annotations

from typing import Any, Callable, Dict, Type

TOOL_REGISTRY: Dict[str, Type[Any]] = {}
AGENT_REGISTRY: Dict[str, Type[Any]] = {}


def register_tool(name: str | None = None) -> Callable[[Any], Any]:
    """Decorator to register a tool class or function.

    Parameters
    ----------
    name : str, optional
        Name under which the tool will be registered. If omitted, ``cls.__name__``
        is used.
    """

    def decorator(obj: Any) -> Any:
        registry_name = name or getattr(obj, "name", getattr(obj, "__name__", str(obj)))
        if registry_name in TOOL_REGISTRY:
            raise ValueError(f"Tool '{registry_name}' already registered")
        TOOL_REGISTRY[registry_name] = obj
        return obj

    return decorator


def get_tool(name: str) -> Type[Any]:
    """Retrieve a registered tool class by name."""
    return TOOL_REGISTRY[name]


def register_agent(name: str | None = None) -> Callable[[Type[Any]], Type[Any]]:
    """Decorator to register an agent class."""

    def decorator(cls: Type[Any]) -> Type[Any]:
        registry_name = name or getattr(cls, "name", cls.__name__)
        if registry_name in AGENT_REGISTRY:
            raise ValueError(f"Agent '{registry_name}' already registered")
        AGENT_REGISTRY[registry_name] = cls
        return cls

    return decorator


def get_agent(name: str) -> Type[Any]:
    """Retrieve a registered agent class by name."""
    return AGENT_REGISTRY[name]

