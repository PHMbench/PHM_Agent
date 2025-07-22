"""Entry point demonstrating a multi-agent workflow with optional instrumentation."""

from __future__ import annotations

import argparse


from agents_config import create_manager_agent
from model_config import configure_model

from smolagents import GradioUI
from dotenv import load_dotenv

load_dotenv(override=True)

# # --- 调试代码开始 ---

# import os # <--- 添加导入
# import sys # <--- 添加导入
# api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
# if not api_key:
#     print("错误：未能在环境变量中找到 GEMINI_API_KEY 或 GOOGLE_API_KEY。", file=sys.stderr)
#     sys.exit(1)
# else:
#     print("成功加载 API 密钥。")
# # --- 调试代码结束 ---


def main(inference: str = "inference_client", inspect: bool = False, ui: bool = False) -> None:
    """Run a demo workflow using smolagents.

    Parameters
    ----------
    inference:
        Inference backend name accepted by :func:`configure_model`.
    inspect:
        Whether to enable OpenInference tracing.
    ui:
        Launch a Gradio web UI instead of running a single query.
    """
    if inspect:
        from openinference.instrumentation.smolagents import SmolagentsInstrumentor
        from phoenix.otel import register

        register()
        SmolagentsInstrumentor().instrument(skip_dep_check=True)

    model = configure_model(inference)
    manager_agent = create_manager_agent(model)

    if ui:
        GradioUI(manager_agent).launch()
        return

    run_result = manager_agent.run(
        "If the US keeps it 2024 growth rate, how many years would it take for the GDP to double?"
    )
    print("Here is the token usage for the manager agent", run_result.token_usage)
    print("Here are the timing informations for the manager agent:", run_result.timing)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the PHM agent demo")
    parser.add_argument(
        "--inference",
        default="litellm",
        choices=["inference_client", "transformers", "ollama", "litellm", "openai"],
        help="Select inference backend",
    )
    parser.add_argument(
        "--inspect", action="store_false", help="Enable OpenInference instrumentation"
    )
    parser.add_argument(
        "--ui", action="store_false", help="Launch a Gradio web UI"
    )
    args = parser.parse_args()
    main(args.inference, args.inspect, args.ui)
