from __future__ import annotations

"""用于构建和分析设备健康指数（Health Index, HI）的工具集。"""

from typing import Dict, List

import numpy as np
from sklearn.decomposition import PCA
from smolagents import tool

from utils.registry import register_tool


@register_tool("create_health_index")
@tool
def create_health_index(features: Dict[str, List[float]], method: str = "pca") -> Dict[str, object]:
    """将诊断Agent输出的多维特征融合成一维的健康指数(HI)。

    Args:
        features: 从诊断Agent获取的特征数据，例如 ``{"rms": [...], "kurtosis": [...]}"。
        method: 使用的融合算法，可选 ``'pca'`` 或 ``'autoencoder'``。

    Returns:
        包含HI时间序列及其元数据的字典，例如 ``{"hi_series": [...], "method": "pca"}``。
    """
    if not features:
        return {"hi_series": [], "method": method}

    data = np.column_stack(list(features.values()))
    if method == "pca":
        hi = PCA(n_components=1).fit_transform(data).ravel()
    elif method == "autoencoder":
        hi = data.mean(axis=1)
    else:
        raise ValueError(f"Unsupported method '{method}'")
    return {"hi_series": hi.tolist(), "method": method}


@register_tool("fit_degradation_model")
@tool
def fit_degradation_model(hi_series: List[float], model_type: str = "exponential") -> Dict[str, object]:
    """对健康指数序列拟合退化模型。

    Args:
        hi_series: 一维的健康指数时间序列。
        model_type: 拟合模型类型，可选 ``'linear'`` 或 ``'exponential'``。

    Returns:
        包含模型类型和关键参数的字典，例如 ``{"model_type": "exponential", "decay_rate": -0.05}``。
    """
    if not hi_series:
        raise ValueError("hi_series cannot be empty")

    t = np.arange(len(hi_series))
    y = np.array(hi_series, dtype=float)

    if model_type == "linear":
        slope, intercept = np.polyfit(t, y, 1)
        return {
            "model_type": "linear",
            "slope": float(slope),
            "intercept": float(intercept),
            "history_length": len(hi_series),
        }

    if model_type == "exponential":
        y_clipped = np.clip(y, 1e-9, None)
        b, a = np.polyfit(t, np.log(y_clipped), 1)
        return {
            "model_type": "exponential",
            "initial": float(np.exp(a)),
            "decay_rate": float(b),
            "history_length": len(hi_series),
        }

    raise ValueError(f"Unsupported model_type '{model_type}'")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    features = {
        "rms": rng.normal(1.0, 0.1, 50).tolist(),
        "kurtosis": rng.normal(3.0, 0.2, 50).tolist(),
    }
    hi = create_health_index(features)
    model = fit_degradation_model(hi["hi_series"], model_type="linear")
    print("Sample HI:", hi)
    print("Fitted model:", model)
