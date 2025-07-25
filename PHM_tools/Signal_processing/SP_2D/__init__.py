"""2-D signal processing tools."""

from .signal_processing_2d import (
    spectrogram,
    mel_spectrogram,
    scalogram,
    gramian_angular_field,
    markov_transition_field,
    recurrence_plot,
    cepstrogram,
    envelope_spectrogram,
)

__all__ = [
    "spectrogram",
    "mel_spectrogram",
    "scalogram",
    "gramian_angular_field",
    "markov_transition_field",
    "recurrence_plot",
    "cepstrogram",
    "envelope_spectrogram",
]


if __name__ == "__main__":
    import numpy as np

    sig = np.random.randn(256)
    spec, _ = spectrogram(sig, fs=1000)
    scal = scalogram(sig, fs=1000)
    print("Spectrogram shape:", spec.shape)
    print("Scalogram shape:", scal.shape)
