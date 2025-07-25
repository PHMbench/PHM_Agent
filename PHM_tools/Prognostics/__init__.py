"""Tools for health index generation and RUL prediction."""

from .health_indicators import create_health_index, fit_degradation_model
from .rul_predictors import predict_rul_from_model

__all__ = [
    "create_health_index",
    "fit_degradation_model",
    "predict_rul_from_model",
]


if __name__ == "__main__":
    import numpy as np

    features = {"rms": np.random.rand(10).tolist(), "kurtosis": np.random.rand(10).tolist()}
    hi = create_health_index(features)
    model = fit_degradation_model(hi["hi_series"], model_type="linear")
    rul = predict_rul_from_model(model, failure_threshold=0.1)
    print("HI sample len:", len(hi["hi_series"]))
    print("Predicted RUL:", rul)
