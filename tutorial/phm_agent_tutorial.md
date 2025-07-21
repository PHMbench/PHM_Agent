# Building a PHM Agent

This tutorial walks through a minimal workflow for analysing vibration signals.
It covers model configuration, creating a small synthetic dataset, building a
local knowledge base, and running the agent.

1. **Prepare a model** using `model_config.configure_model()`.
2. **Create the benchmark dataset** and load a signal:
   ```python
   from benchmark import BenchmarkDataset, create_example_dataset

   metadata = create_example_dataset('tmp/data')
   dataset = BenchmarkDataset(metadata)
   signal = dataset.load(0)
   ```
3. **Build a retriever** from your own documents:
   ```python
   from PHM_tools.Retrieval.local_knowledge import create_local_retriever_tool
   retriever = create_local_retriever_tool('knowledge_base')
   ```
4. **Instantiate the agent** with the required tools and model. Include the retriever and any
   signal-processing tools you need.
5. **Run analysis** by providing the loaded signal to the agent and asking your question.

The agent will combine the knowledge base with its tools to analyse the signal and return an answer.

Check `demo/existing_data_demo.py` for a minimal runnable example.
