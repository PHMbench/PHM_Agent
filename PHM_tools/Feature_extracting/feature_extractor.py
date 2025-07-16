from __future__ import annotations

"""Signal feature extraction tools used by the PHM agent."""

from typing import Dict

import numpy as np
import scipy.signal
import scipy.stats
from smolagents import tool


def _ensure_3d(x: np.ndarray) -> np.ndarray:
    """Ensure input has shape ``(B, L, C)``.

    If ``x`` is 1D, it becomes ``(1, L, 1)``.
    If ``x`` is 2D, it becomes ``(B, L, 1)``.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        return x[None, :, None]
    if x.ndim == 2:
        return x[:, :, None]
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected 1D, 2D, or 3D array, got shape {x.shape}")

from utils.registry import register_tool


@register_tool("FeatureExtractorTools")
class FeatureExtractorTools:
    """Collection of functions to compute common health indicators.

    All methods accept arrays with shape ``(B, L, C)`` representing a batch of
    multichannel signals. Inputs with shape ``(B, L)`` or ``(L,)`` are
    automatically expanded so that the output tensors always follow the
    ``(B, L, C)`` convention. Features are computed along the length dimension
    for every batch and channel. NaN or infinity values are replaced with ``0``
    for robustness.
    """

    TIME_FEATURES = [
        "Mean",
        "Std",
        "Var",
        "Entropy",
        "Max",
        "Min",
        "AbsMean",
        "Kurtosis",
        "RMS",
        "Median",
        "MAD",
        "CrestFactor",
        "ClearanceFactor",
        "Skewness",
        "ShapeFactor",
        "CrestFactorDelta",
        "PeakToPeak",
        "ImpulseFactor",
        "MarginFactor",
    ]

    FREQ_FEATURES = [
        "EnvelopeMean",
        "EnvelopeStd",
        "TKE",
        "SpectralCentroid",
        "SpectralRolloff",
        "SpectralFlatness",
        "SpectralEntropy",
        "SpectralSlope",
    ]

    def extract_time_features(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract a set of time-domain features from ``signal``.

        Parameters
        ----------
        signal:
            Array with shape ``(B, L, C)`` or smaller.

        Returns
        -------
        dict
            Mapping from feature name to array of shape ``(B, C)``.
        """
        print("  TOOLBOX: Extracting time-domain features...")
        x = _ensure_3d(signal)
        if x.size == 0:
            return {}

        B, L, C = x.shape

        mu = x.mean(axis=1)
        std = x.std(axis=1)
        var = x.var(axis=1)
        abs_x = np.abs(x)
        abs_mean = abs_x.mean(axis=1)
        rms = np.sqrt(np.mean(x**2, axis=1))
        median = np.median(x, axis=1)
        mad = np.median(np.abs(x - median[:, None, :]), axis=1)
        diff = np.diff(x, axis=1)

        hist = np.histogram(x, bins=10, axis=1)[0]
        entropy = scipy.stats.entropy(np.abs(hist), axis=1)
        kurtosis = scipy.stats.kurtosis(x, axis=1)
        skewness = scipy.stats.skew(x, axis=1)
        max_val = x.max(axis=1)
        min_val = x.min(axis=1)
        crest_factor_delta = np.sqrt(np.mean(diff**2, axis=1))

        rms_safe = np.where(rms > 0, rms, 1e-9)
        abs_mean_safe = np.where(abs_mean > 0, abs_mean, 1e-9)

        features = {
            "Mean": mu,
            "Std": std,
            "Var": var,
            "Entropy": entropy,
            "Max": max_val,
            "Min": min_val,
            "AbsMean": abs_mean,
            "Kurtosis": kurtosis,
            "RMS": rms,
            "Median": median,
            "MAD": mad,
            "CrestFactor": np.max(abs_x, axis=1) / rms_safe,
            "ClearanceFactor": np.max(abs_x, axis=1) / abs_mean_safe,
            "Skewness": skewness,
            "ShapeFactor": rms / abs_mean_safe,
            "CrestFactorDelta": crest_factor_delta / abs_mean_safe,
            "PeakToPeak": max_val - min_val,
            "ImpulseFactor": np.max(abs_x, axis=1) / abs_mean_safe,
            "MarginFactor": np.max(abs_x, axis=1)
            / (np.mean(np.sqrt(abs_x), axis=1) ** 2 + 1e-9),
        }

        out = {k: v.reshape(B, C) for k, v in features.items()}
        for k, v in out.items():
            v[np.isnan(v) | np.isinf(v)] = 0.0
        return out














    def extract_frequency_features(
        self, signal: np.ndarray, fs: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """Extract advanced frequency-domain features.

        Parameters
        ----------
        signal:
            Array with shape ``(B, L, C)`` or smaller.
        fs:
            Sampling frequency. Defaults to ``1.0``.

        Returns
        -------
        dict
            Mapping from feature name to array of shape ``(B, C)``.
        """
        print("  TOOLBOX: Extracting advanced features...")
        x = _ensure_3d(signal)
        if x.size == 0:
            return {}

        B, L, C = x.shape

        analytic = scipy.signal.hilbert(x, axis=1)
        envelope = np.abs(analytic)
        if L > 2:
            tke = np.mean(x[:, 1:-1, :] ** 2 - x[:, :-2, :] * x[:, 2:, :], axis=1)
        else:
            tke = np.zeros((B, C))

        f, Pxx = scipy.signal.periodogram(x, fs=fs, axis=1)
        psd_sum = np.sum(Pxx, axis=1) + 1e-12
        spectral_centroid = np.sum(Pxx * f[None, :, None], axis=1) / psd_sum
        cumulative = np.cumsum(Pxx, axis=1)
        idx = np.argmax(cumulative >= 0.85 * psd_sum[:, None, :], axis=1)
        spectral_rolloff = f[idx]
        spectral_flatness = scipy.stats.gmean(Pxx + 1e-12, axis=1) / (np.mean(Pxx, axis=1) + 1e-12)
        psd_norm = Pxx / psd_sum[:, None, :]
        spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12), axis=1)
        mean_f = f.mean()
        var_f = np.mean((f - mean_f) ** 2)
        mean_psd = np.mean(Pxx, axis=1, keepdims=True)
        spectral_slope = np.sum((f - mean_f) * (Pxx - mean_psd), axis=1) / (
            Pxx.shape[1] * var_f
        )

        features = {
            "EnvelopeMean": envelope.mean(axis=1),
            "EnvelopeStd": envelope.std(axis=1),
            "TKE": tke,
            "SpectralCentroid": spectral_centroid,
            "SpectralRolloff": spectral_rolloff,
            "SpectralFlatness": spectral_flatness,
            "SpectralEntropy": spectral_entropy,
            "SpectralSlope": spectral_slope,
        }

        out = {k: v.reshape(B, C) for k, v in features.items()}
        for k, v in out.items():
            v[np.isnan(v) | np.isinf(v)] = 0.0
        return out


# ---------------------------------------------------------------------------
# Tool wrappers
# ---------------------------------------------------------------------------

@register_tool("extract_time_features")
@tool
def extract_time_features_tool(signal: list | np.ndarray) -> Dict[str, np.ndarray]:
    """Wrapper around :class:`FeatureExtractorTools.extract_time_features`.

    Args:
        signal: Input signal as a list or NumPy array.
    """
    return FeatureExtractorTools().extract_time_features(np.asarray(signal))


@register_tool("extract_frequency_features")
@tool
def extract_frequency_features_tool(
    signal: list | np.ndarray, fs: float = 1.0
) -> Dict[str, np.ndarray]:
    """Wrapper around :class:`FeatureExtractorTools.extract_frequency_features`.

    Args:
        signal: Input signal as a list or NumPy array.
        fs: Sampling frequency. Defaults to ``1.0``.
    """
    return FeatureExtractorTools().extract_frequency_features(np.asarray(signal), fs=fs)
