"""Anomaly detection utilities for PHM_Agent."""

from .signal_comparison import calculate_statistical_divergence

__all__ = ["calculate_statistical_divergence"]


if __name__ == "__main__":
    import numpy as np

    a = np.random.randn(100)
    b = np.random.randn(100)
    from .signal_comparison import calculate_statistical_divergence

    res = calculate_statistical_divergence(a.tolist(), b.tolist())
    print("Divergence sample:", res)
