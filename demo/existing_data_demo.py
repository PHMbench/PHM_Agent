"""Demo showing how to load benchmark data and run retrieval."""

from __future__ import annotations

from smolagents import CodeAgent, LiteLLMModel

from benchmark import BenchmarkDataset
from PHM_tools.Retrieval.local_knowledge import create_local_retriever_tool
from pathlib import Path
import subprocess
from utils.registry import get_tool


def main() -> None:
    model = LiteLLMModel(model_id="gpt-4o")
    data_path = Path("benchmark/data/0.h5")
    if not data_path.exists():
        subprocess.run(["python", "benchmark/generate_dummy_data.py"], check=True)
    dataset = BenchmarkDataset("benchmark/data/metadata.csv")
    signal = dataset.load(0)
    retriever = create_local_retriever_tool("knowledge_base")

    time_feat = get_tool("extract_time_features")
    agent = CodeAgent(
        tools=[retriever, time_feat()],
        model=model,
        max_steps=4,
        verbosity_level=2,
        stream_outputs=True,
    )
    result = agent.run("Describe the loaded signal")
    print(result)


if __name__ == "__main__":
    main()
