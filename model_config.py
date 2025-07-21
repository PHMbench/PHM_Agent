"""Model configuration helpers following smolagents examples."""

from typing import Literal
from dotenv import load_dotenv

load_dotenv(override=True)
from smolagents import (
    InferenceClientModel,
    LiteLLMModel,
    OpenAIServerModel,
    TransformersModel,
    ToolCallingAgent,
    tool,
)
from smolagents.models import Model


def configure_model(
    inference: Literal[
        "inference_client",
        "transformers",
        "ollama",
        "litellm",
        "openai",
    ] = "inference_client",
) -> Model:
    """Return a model instance for the chosen inference backend."""
    if inference == "inference_client":
        return InferenceClientModel(model_id="meta-llama/Llama-3.3-70B-Instruct", provider="nebius")
    if inference == "transformers":
        return TransformersModel(model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct", device_map="auto", max_new_tokens=1000)
    if inference == "ollama":
        return LiteLLMModel(
            model_id="ollama_chat/llama3.2",
            api_base="http://localhost:11434",
            api_key="your-api-key",
            num_ctx=8192,
        )
    if inference == "litellm":
        return LiteLLMModel(model_id="gemini")
    if inference == "openai":
        return OpenAIServerModel(model_id="gpt-4o")
    raise ValueError(f"Unknown inference type: {inference}")


if __name__ == "__main__":
    model = configure_model()

    @tool
    def dummy_tool(x: int) -> int:
        """A minimal example tool."""
        return x + 1

    agent = ToolCallingAgent(tools=[dummy_tool], model=model, verbosity_level=2)
    print("ToolCallingAgent demo:", agent.run("Use dummy_tool on number 1"))
