from __future__ import annotations

"""用于预测设备剩余使用寿命（RUL）的工具集。"""

import math
from typing import Dict

from smolagents import tool
from utils.registry import register_tool


@register_tool("predict_rul_from_model")
@tool
def predict_rul_from_model(degradation_model: Dict[str, object], failure_threshold: float) -> Dict[str, float | list]:
    """基于退化模型外推失效时间并计算RUL。

    Args:
        degradation_model: :func:`fit_degradation_model` 返回的模型参数。
        failure_threshold: 定义设备失效的健康指数阈值。

    Returns:
        包含RUL估计值和置信区间的字典，例如 ``{"predicted_rul": 150.5, "confidence_interval": [120.0, 180.0]}``。
    """
    model_type = degradation_model.get("model_type")
    history_len = degradation_model.get("history_length", 0)

    if model_type == "linear":
        slope = float(degradation_model["slope"])
        intercept = float(degradation_model["intercept"])
        if slope >= 0:
            raise ValueError("Linear degradation slope must be negative")
        t_failure = (failure_threshold - intercept) / slope
    elif model_type == "exponential":
        initial = float(degradation_model["initial"])
        decay_rate = float(degradation_model["decay_rate"])
        if decay_rate >= 0:
            raise ValueError("Decay rate must be negative for degradation")
        t_failure = math.log(failure_threshold / initial) / decay_rate
    else:
        raise ValueError(f"Unsupported model type '{model_type}'")

    rul = t_failure - history_len
    ci = [rul * 0.9, rul * 1.1]
    return {"predicted_rul": float(rul), "confidence_interval": [float(ci[0]), float(ci[1])]}


if __name__ == "__main__":
    model = {"model_type": "linear", "slope": -0.01, "intercept": 1.0, "history_length": 100}
    result = predict_rul_from_model(model, failure_threshold=0.2)
    print("Sample RUL prediction:", result)
