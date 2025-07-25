"""Fault diagnosis utilities for PHM_Agent."""

from .classifiers import classify_fault_from_features

__all__ = ["classify_fault_from_features"]


if __name__ == "__main__":
    sample = {"rms": 0.4, "kurtosis": 3.1}
    from .classifiers import classify_fault_from_features

    res = classify_fault_from_features(sample)
    print("Classification sample:", res)
