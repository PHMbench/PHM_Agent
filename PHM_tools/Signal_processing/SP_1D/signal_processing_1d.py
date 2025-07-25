from __future__ import annotations

"""Basic 1-D preprocessing utilities."""


import numpy as np
import scipy.signal
from smolagents import tool

from utils import ensure_3d
from utils.registry import register_tool


@register_tool("normalize")
@tool
def normalize(signal: list | np.ndarray) -> np.ndarray:
    """Standardize a vibration signal.

    Args:
        signal: Signal to normalize.

    Returns:
        The normalized signal with shape ``(B, L, C)``.
    """
    x = ensure_3d(np.asarray(signal))
    mean = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True) + 1e-9
    return (x - mean) / std


@register_tool("detrend")
@tool
def detrend(signal: list | np.ndarray) -> np.ndarray:
    """Remove the linear trend from a signal.

    Args:
        signal: Input vibration signal.

    Returns:
        The detrended signal with shape ``(B, L, C)``.
    """
    x = ensure_3d(np.asarray(signal))
    return scipy.signal.detrend(x, axis=1)


@register_tool("bandpass")
@tool
def bandpass(
    signal: list | np.ndarray,
    fs: float,
    low: float,
    high: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter.

    Args:
        signal: Input vibration signal.
        fs: Sampling frequency.
        low: Low cutoff frequency in hertz.
        high: High cutoff frequency in hertz.
        order: Filter order. Defaults to ``4``.

    Returns:
        The band-pass filtered signal with shape ``(B, L, C)``.
    """
    x = ensure_3d(np.asarray(signal))
    b, a = scipy.signal.butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    return scipy.signal.filtfilt(b, a, x, axis=1)


@register_tool("fft")
@tool
def fft(signal: list | np.ndarray) -> np.ndarray:
    """Return the magnitude FFT of a signal.

    Args:
        signal: Signal for the transform.

    Returns:
        Magnitude spectrum with shape ``(B, F, C)``.
    """
    x = ensure_3d(np.asarray(signal))
    spec = np.fft.rfft(x, axis=1)
    return np.abs(spec)


@register_tool("cepstrum")
@tool
def cepstrum(signal: list | np.ndarray) -> np.ndarray:
    """Compute the real cepstrum of a signal.

    Args:
        signal: Signal for the transform.

    Returns:
        Cepstrum with shape ``(B, L, C)``.
    """
    spectrum = fft(signal)
    log_mag = np.log(spectrum + 1e-9)
    return np.fft.irfft(log_mag, axis=1)


@register_tool("envelope_spectrum")
@tool
def envelope_spectrum(signal: list | np.ndarray) -> np.ndarray:
    """Compute the spectrum of a signal's Hilbert envelope.

    Args:
        signal: Input vibration signal.

    Returns:
        Magnitude spectrum of the envelope with shape ``(B, F, C)``.
    """
    x = ensure_3d(np.asarray(signal))
    analytic = scipy.signal.hilbert(x, axis=1)
    env = np.abs(analytic)
    return fft(env)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sig = rng.normal(size=1024)
    filtered = bandpass(sig, fs=1000, low=5, high=200)
    spec = fft(filtered)
    env_spec = envelope_spectrum(filtered)
    print("FFT result shape:", spec.shape)
    print("Envelope spectrum shape:", env_spec.shape)
