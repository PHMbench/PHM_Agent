# PHM Agent Usage

This repository demonstrates a lightweight workflow for building predictive health management agents using the [smolagents](https://github.com/huggingface/smolagents) framework. The `PHMAgent` class mirrors the examples in `smolagents-referance/examples` and can be extended with custom tools.

## Running a Demo

Refer to the demos under `smolagents-referance/examples` for guidance on creating a tool-calling agent. A minimal script might look like:

```python
from smolagents import InferenceClientModel
from utils.registry import get_agent, get_tool

model = InferenceClientModel()
FeatureTools = get_tool("FeatureExtractorTools")
SignalTools = get_tool("SignalProcessingTools")
AgentCls = get_agent("PHMAgent")

agent = AgentCls(tools=[FeatureTools(), SignalTools()], model=model)
```

All tools operate on `(B, L, C)` arrays, automatically expanding lower-dimensional inputs. The
`model_config.py` module exposes a `configure_model()` function which returns a model instance based on
the selected inference backend. Use it to create agents consistent with the demos under
`smolagents-referance/examples`.
