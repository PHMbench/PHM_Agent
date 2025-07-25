"""1-D signal processing tools."""

from .signal_processing_1d import (
    normalize,
    detrend,
    bandpass,
    fft,
    cepstrum,
    envelope_spectrum,
)

__all__ = [
    "normalize",
    "detrend",
    "bandpass",
    "fft",
    "cepstrum",
    "envelope_spectrum",
]


if __name__ == "__main__":
    import numpy as np

    sig = np.random.randn(500)
    filtered = bandpass(sig, fs=1000, low=10, high=200)
    spec = fft(filtered)
    print("Bandpass sample shape:", filtered.shape)
    print("FFT sample shape:", spec.shape)
