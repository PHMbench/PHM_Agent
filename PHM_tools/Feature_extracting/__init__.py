"""Feature extraction tools for PHM_Agent."""

from .feature_extractor import extract_time_features, extract_frequency_features

__all__ = ["extract_time_features", "extract_frequency_features"]


if __name__ == "__main__":
    import numpy as np

    signal = np.random.randn(1_000)
    time_feats = extract_time_features(signal)
    freq_feats = extract_frequency_features(signal, fs=1000.0)
    print("Time feature keys:", list(time_feats.keys())[:3])
    print("Frequency feature keys:", list(freq_feats.keys())[:3])
