"""Signal preprocessing tools for PHM_Agent."""

from .SP_1D import (
    normalize,
    detrend,
    bandpass,
    fft,
    cepstrum,
    envelope_spectrum,
)
from .SP_2D import (
    spectrogram,
    mel_spectrogram,
    scalogram,
    gramian_angular_field,
    markov_transition_field,
    recurrence_plot,
    cepstrogram,
    envelope_spectrogram,
)
from .slicer_tools import apply_transform, extract_slice

__all__ = [
    "normalize",
    "detrend",
    "bandpass",
    "fft",
    "cepstrum",
    "envelope_spectrum",
    "spectrogram",
    "mel_spectrogram",
    "scalogram",
    "gramian_angular_field",
    "markov_transition_field",
    "recurrence_plot",
    "cepstrogram",
    "envelope_spectrogram",
    "apply_transform",
    "extract_slice",
]


if __name__ == "__main__":
    import numpy as np

    signal = np.random.randn(1024)
    norm = normalize(signal)
    spec = spectrogram(signal, fs=1000)[0]
    print("Normalized sample shape:", norm.shape)
    print("Spectrogram sample shape:", spec.shape)
