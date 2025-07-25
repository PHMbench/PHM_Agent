from __future__ import annotations

"""直接在信号层面对比，用于检测细微或非平稳变化的工具。"""

from typing import Dict, List

import numpy as np
from smolagents import tool

from utils.registry import register_tool


@register_tool("calculate_statistical_divergence")
@tool
def calculate_statistical_divergence(
    series_a: List[float], series_b: List[float], method: str = "kl"
) -> Dict[str, float | str]:
    """计算两个信号分布的差异以检测状态变化。

    Args:
        series_a: 在线信号序列。
        series_b: 参考信号序列。
        method: 散度计算方法, ``'kl'``或 ``'js'``。

    Returns:
        包含散度值和所用方法的字典, 例如 ``{"divergence": 0.08, "method": "kl"}``。
    """
    a = np.asarray(series_a, dtype=float)
    b = np.asarray(series_b, dtype=float)
    hist_a, _ = np.histogram(a, bins=100, density=True)
    hist_b, _ = np.histogram(b, bins=100, density=True)
    hist_a += 1e-12
    hist_b += 1e-12
    p = hist_a / hist_a.sum()
    q = hist_b / hist_b.sum()

    if method == "kl":
        divergence = float(np.sum(p * np.log(p / q)))
    elif method == "js":
        m = 0.5 * (p + q)
        divergence = float(
            0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
        )
    else:
        raise ValueError(f"Unsupported method '{method}'")

    return {"divergence": divergence, "method": method}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    series_a = rng.normal(0, 1, 1000).tolist()
    series_b = rng.normal(0.1, 1.2, 1000).tolist()
    result = calculate_statistical_divergence(series_a, series_b)
    print("Sample divergence:", result)
