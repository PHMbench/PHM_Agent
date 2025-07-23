# PHM_Agent

This repository provides building blocks for Predictive Health Management agents
leveraging the `smolagents` framework.

## Registration System

Tools and agents are registered with simple decorators for easy discovery.

```python
from utils.registry import register_tool, register_agent

@register_tool("extract_time_features")
def extract_time_features(signal: list[float]) -> dict:
    ...
@register_agent("PHMAgent")
class PHMAgent(ToolCallingAgent):
    ...
```

Registered components can later be retrieved via `get_tool` or `get_agent` from
`utils.registry`.

All preprocessing and feature extraction utilities operate on arrays shaped as
`(B, L, C)` where `B` is batch size, `L` is signal length and `C` is the number
of channels. Inputs with fewer dimensions are automatically expanded so that a
1‑D array becomes `(1, L, 1)`.
Feature extraction functions compute statistics directly across the batch and
channel dimensions without reshaping the input.

## Signal Processing Functions

The signal processing module exposes a number of 1‑D and 2‑D preprocessing utilities
often used in PHM research. The most notable ones are:

- `normalize`, `detrend`, `bandpass`
- `fft`, `cepstrum`, `envelope_spectrum`
- `spectrogram`, `mel_spectrogram`, `scalogram`
- `gramian_angular_field`, `markov_transition_field`, `recurrence_plot`
- `cepstrogram`, `envelope_spectrogram`

## Decision-Making Tools

Simple machine-learning helpers are provided for anomaly detection and fault
diagnosis:

- `isolation_forest_detector` for unsupervised anomaly detection
- `svm_fault_classifier` for supervised fault classification

Each function accepts arrays with shape `(B, L, C)` and returns either a
processed signal of the same shape or a 2‑D representation of shape
`(B, *, *, C)` depending on the method.

Each exposed function is decorated with `@tool` from `smolagents` so that it can
be called by a `ToolCallingAgent`. See `model_config.py` for a minimal example
showing how to instantiate a model and agent using these tools.

`RetrieverTool` is provided for semantic search over documentation using a
Chroma vector store. Build the store with
`PHM_tools.Retrieval.build_vector_store()` and pass a `RetrieverTool` instance to
an agent.

The repository also exposes `model_download_tool` which returns the Hugging Face
model with the highest download count for a given pipeline task.

## Demo

Run `main.py` to launch a small multi-agent workflow. The script mirrors the
`inspect_multiagent_run.py` example from `smolagents-referance` and accepts the
same inference backends as `model_config.configure_model`.

```bash
python main.py --inspect
```

The `--inspect` flag enables OpenInference tracing via Phoenix so that you can
observe detailed metrics about each agent run.

To experiment interactively, launch the Gradio interface:

```bash
python -m UI.gradio_app
```


## Agent Configuration

`agents_config.py` exposes a `create_manager_agent()` helper that instantiates a
manager agent together with a search agent and a PHM analysis agent. Both
sub-agents make use of registered tools so you can swap in your own
implementations via the registry.

```python
from agents_config import create_manager_agent
from model_config import configure_model

model = configure_model()
agent = create_manager_agent(model)
agent.run("What's the weather in Paris?")
```

### Gradio UI

`demo/gradio_app.py` launches a simple web interface allowing you to chat with
the manager agent. The interface mirrors the example from
`smolagents-referance` and lets the agent decide which tool to use for each
query.

```bash
python demo/gradio_app.py
```
## Retrieval-Augmented Generation

`RetrieverTool` provides semantic search over a small documentation corpus using
Chroma. Use `build_vector_store()` to create the vector store and pass it to the
tool when constructing an agent.

```python
from PHM_tools.RAG import build_vector_store, RetrieverTool

vector_store = build_vector_store()
retriever = RetrieverTool(vector_store)
```

The tool is registered and can be obtained via `get_tool("RetrieverTool")`.

## Benchmark Dataset and Local Knowledge Base

`benchmark` provides utilities to create and load a small demo dataset stored as
HDF5 files. First generate the data:

```bash
python benchmark/generate_dummy_data.py
```

Then use `BenchmarkDataset` to access the files by ID:

```python
from benchmark import BenchmarkDataset
dataset = BenchmarkDataset("benchmark/data/metadata.csv")
sample = dataset.load(0)
metadata = dataset.metadata
```

For retrieval over your own PDF or Markdown documents, build a local vector
store and tool:

```python
from PHM_tools.Retrieval import create_local_retriever_tool
retriever = create_local_retriever_tool("knowledge_base")
```

See `tutorial/phm_agent_tutorial.md` for a short walkthrough and
`demo/existing_data_demo.py` for a runnable example combining both utilities.

## Configuration

Application behaviour is controlled via `config.yaml`. The file is parsed into a
small `Config` dataclass (see `utils/config.py`) so fields can be accessed as
attributes. In addition to selecting the inference backend and model
identifier, you can list which sub-agents the manager should load:

```yaml
inference: litellm
model_id: gemini/gemini-2.5-pro
enabled_agents:
  - search_agent
  - phm_agent
  - retrieval_agent
  - deep_research_agent
```

Add or remove agent names from `enabled_agents` to customise the workflow. For
example, including `deep_research_agent` enables an advanced helper that
performs in-depth web searches before handing results to the PHM agent.

### Building the Documentation

This repository ships with a minimal Sphinx setup under `docs/`. Install the
documentation dependencies and build the HTML pages with:

```bash
pip install -r docs/requirements.txt
make -C docs html
```

Documentation is automatically deployed to GitHub Pages whenever the `main`
branch is updated.
