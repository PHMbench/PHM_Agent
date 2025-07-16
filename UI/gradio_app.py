"""Launch the PHM agent with a Gradio interface."""

from smolagents import CodeAgent, GradioUI, WebSearchTool

from agents_config import create_manager_agent
from model_config import configure_model


def main(inference: str = "inference_client") -> None:
    model = configure_model(inference)
    agent = create_manager_agent(model)
    demo_agent = CodeAgent(
        tools=[WebSearchTool()],
        model=model,
        managed_agents=[agent],
        verbosity_level=1,
        planning_interval=3,
        stream_outputs=True,
        name="phm_demo",
        description="PHM manager agent with web search",
    )
    GradioUI(demo_agent, file_upload_folder="./data").launch()


if __name__ == "__main__":
    main()
