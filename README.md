# PHM_Agent

This repository provides building blocks for Predictive Health Management agents
leveraging the `smolagents` framework.

## Registration System

Tools and agents are registered with simple decorators for easy discovery.

```python
from utils.registry import register_tool, register_agent

@register_tool("FeatureExtractorTools")
class FeatureExtractorTools:
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

`SignalProcessingTools` exposes a number of 1‑D and 2‑D preprocessing utilities
often used in PHM research. The most notable ones are:

- `normalize`, `detrend`, `bandpass`
- `fft`, `cepstrum`, `envelope_spectrum`
- `spectrogram`, `mel_spectrogram`, `scalogram`
- `gramian_angular_field`, `markov_transition_field`, `recurrence_plot`
- `cepstrogram`, `envelope_spectrogram`

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
