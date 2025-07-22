# Building a PHM Agent

This tutorial guides you through a minimal workflow for analysing the demo
dataset and querying a small knowledge base.

1. **Generate sample data**:
   ```bash
   python benchmark/generate_dummy_data.py
   ```
2. **Prepare a model** using `model_config.configure_model()`.
3. **Load a signal** using `BenchmarkDataset`:
   ```python
   from benchmark import BenchmarkDataset
   dataset = BenchmarkDataset("benchmark/data/metadata.csv")
   signal = dataset.load(0)
   ```
4. **Build a retriever** from your own documents:
   ```python
   from PHM_tools.Retrieval import create_local_retriever_tool
   retriever = create_local_retriever_tool("knowledge_base")
   ```
5. **Instantiate the agent** with the retriever and the desired tools.
6. **Run analysis** by providing the loaded signal and asking your question.

The agent combines the knowledge base with signal-processing tools to analyse
the data and return an answer. Check `demo/existing_data_demo.py` for a minimal
runnable example.
