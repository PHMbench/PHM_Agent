from __future__ import annotations

"""2-D representations for vibration signals."""

import numpy as np
import scipy.signal
from smolagents import tool

from utils import ensure_3d
from utils.registry import register_tool


@register_tool("spectrogram")
@tool
def spectrogram(
    signal: list | np.ndarray,
    fs: float,
    nperseg: int = 256,
    noverlap: int | None = None,
) -> np.ndarray:
    """Compute a magnitude spectrogram using STFT.

    Args:
        signal: Input vibration signal.
        fs: Sampling frequency.
        nperseg: Length of each FFT segment.
        noverlap: Number of points to overlap between segments.

    Returns:
        Spectrogram with shape ``(B, F, T, C)``.
    """
    x = ensure_3d(np.asarray(signal))
    x = np.transpose(x, (0, 2, 1))  # B, C, L
    _, _, Zxx = scipy.signal.stft(
        x,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap if noverlap is not None else nperseg // 2,
        axis=2,
    )
    mag = np.abs(Zxx)
    return np.transpose(mag, (0, 2, 3, 1))


def _mel_filterbank(n_fft: int, n_mels: int, fs: float, fmin: float, fmax: float) -> np.ndarray:
    """Create a mel filterbank matrix."""

    def hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10 ** (mel / 2595.0) - 1)

    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / fs).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_m_minus, f_m, f_m_plus = bins[m - 1], bins[m], bins[m + 1]
        for k in range(f_m_minus, f_m):
            fb[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus + 1e-9)
        for k in range(f_m, f_m_plus):
            fb[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m + 1e-9)
    return fb


@register_tool("mel_spectrogram")
@tool
def mel_spectrogram(
    signal: list | np.ndarray,
    fs: float,
    nperseg: int = 256,
    noverlap: int | None = None,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Compute a mel scale spectrogram.

    Args:
        signal: Input vibration signal.
        fs: Sampling frequency.
        nperseg: Length of each FFT segment.
        noverlap: Number of points to overlap between segments.
        n_mels: Number of mel bands.
        fmin: Lowest frequency.
        fmax: Highest frequency. Defaults to ``fs/2`` when ``None``.

    Returns:
        Mel spectrogram with shape ``(B, n_mels, T, C)``.
    """
    spec = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    power = spec**2
    if fmax is None:
        fmax = fs / 2
    fb = _mel_filterbank(nperseg, n_mels, fs, fmin, fmax)
    return np.einsum("bftc,mf->bmtc", power, fb)


@register_tool("scalogram")
@tool
def scalogram(
    signal: list | np.ndarray,
    scales: np.ndarray | None = None,
    wavelet_width: float = 5.0,
) -> np.ndarray:
    """Compute a continuous wavelet transform magnitude image.

    Args:
        signal: Input vibration signal.
        scales: Scales used in the wavelet transform.
        wavelet_width: Width of the Morlet wavelet.

    Returns:
        Scalogram with shape ``(B, len(scales), L, C)``.
    """
    x = ensure_3d(np.asarray(signal))
    if scales is None:
        scales = np.arange(1, 33)
    B, L, C = x.shape
    out = np.empty((B, scales.size, L, C), dtype=float)
    for b in range(B):
        for c in range(C):
            coeff = scipy.signal.cwt(
                x[b, :, c],
                lambda m, s: scipy.signal.morlet2(m, s, w=wavelet_width),
                scales,
            )
            out[b, :, :, c] = np.abs(coeff)
    return out


@register_tool("gramian_angular_field")
@tool
def gramian_angular_field(signal: list | np.ndarray) -> np.ndarray:
    """Compute a Gramian Angular Field image.

    Args:
        signal: Input vibration signal.

    Returns:
        GAF representation with shape ``(B, L, L, C)``.
    """
    x = ensure_3d(np.asarray(signal))
    B, L, C = x.shape
    x_min = x.min(axis=1, keepdims=True)
    x_max = x.max(axis=1, keepdims=True)
    x_scaled = (x - x_min) / (x_max - x_min + 1e-9)
    x_scaled = np.clip(x_scaled, 0, 1)
    phi = np.arccos(2 * x_scaled - 1)
    gaf = np.zeros((B, L, L, C))
    for b in range(B):
        for c in range(C):
            p = phi[b, :, c]
            gaf[b, :, :, c] = np.cos(p[:, None] + p[None, :])
    return gaf


@register_tool("markov_transition_field")
@tool
def markov_transition_field(signal: list | np.ndarray, bins: int = 8) -> np.ndarray:
    """Compute a Markov Transition Field image.

    Args:
        signal: Input vibration signal.
        bins: Number of quantization bins.

    Returns:
        MTF representation with shape ``(B, L, L, C)``.
    """
    x = ensure_3d(np.asarray(signal))
    B, L, C = x.shape
    out = np.zeros((B, L, L, C))
    for b in range(B):
        for c in range(C):
            s = x[b, :, c]
            edges = np.linspace(s.min(), s.max(), bins + 1)
            states = np.digitize(s, edges) - 1
            trans = np.zeros((bins, bins))
            for i in range(L - 1):
                trans[states[i], states[i + 1]] += 1
            trans /= trans.sum(axis=1, keepdims=True) + 1e-9
            for i in range(L):
                for j in range(L):
                    out[b, i, j, c] = trans[states[i], states[j]]
    return out


@register_tool("recurrence_plot")
@tool
def recurrence_plot(signal: list | np.ndarray, eps: float | None = None) -> np.ndarray:
    """Generate a binary recurrence plot.

    Args:
        signal: Input vibration signal.
        eps: Distance threshold for recurrence.

    Returns:
        Recurrence plot with shape ``(B, L, L, C)``.
    """
    x = ensure_3d(np.asarray(signal))
    B, L, C = x.shape
    out = np.empty((B, L, L, C))
    for b in range(B):
        for c in range(C):
            s = x[b, :, c]
            threshold = 0.1 * np.std(s) if eps is None else eps
            dist = np.abs(s[:, None] - s[None, :])
            out[b, :, :, c] = (dist <= threshold).astype(float)
    return out


@register_tool("cepstrogram")
@tool
def cepstrogram(
    signal: list | np.ndarray,
    fs: float,
    nperseg: int = 256,
    noverlap: int | None = None,
) -> np.ndarray:
    """Compute a cepstrum over time using STFT.

    Args:
        signal: Input vibration signal.
        fs: Sampling frequency.
        nperseg: Length of each FFT segment.
        noverlap: Number of points to overlap between segments.

    Returns:
        Cepstrogram with shape ``(B, F, T, C)``.
    """
    spec = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    log_spec = np.log(spec + 1e-9)
    return np.fft.irfft(log_spec, axis=1)


@register_tool("envelope_spectrogram")
@tool
def envelope_spectrogram(
    signal: list | np.ndarray,
    fs: float,
    nperseg: int = 256,
    noverlap: int | None = None,
) -> np.ndarray:
    """Compute a spectrogram of the signal envelope.

    Args:
        signal: Input vibration signal.
        fs: Sampling frequency.
        nperseg: Length of each FFT segment.
        noverlap: Number of points to overlap between segments.

    Returns:
        Spectrogram of the envelope with shape ``(B, F, T, C)``.
    """
    x = ensure_3d(np.asarray(signal))
    analytic = scipy.signal.hilbert(x, axis=1)
    env = np.abs(analytic)
    return spectrogram(env, fs=fs, nperseg=nperseg, noverlap=noverlap)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    signal = rng.normal(size=2048)
    spec = spectrogram(signal, fs=1000)
    mel = mel_spectrogram(signal, fs=1000)
    print("Spectrogram shape:", spec.shape)
    print("Mel-spectrogram shape:", mel.shape)

