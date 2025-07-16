from __future__ import annotations

"""Basic signal preprocessing tools supporting B, L, C shaped arrays."""

from typing import Any, Dict

import numpy as np
import scipy.signal

from smolagents import tool
from utils.registry import register_tool


def _ensure_3d(x: np.ndarray) -> np.ndarray:
    """Ensure input ``x`` has shape (B, L, C).

    If ``x`` is 1-D, it is treated as (L,) and reshaped to (1, L, 1).
    If ``x`` is 2-D, it is treated as (B, L) and reshaped to (B, L, 1).
    """
    x = np.asarray(x)
    if x.ndim == 1:
        return x[None, :, None]
    if x.ndim == 2:
        return x[:, :, None]
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected 1D, 2D, or 3D array, got shape {x.shape}")


@register_tool("SignalProcessingTools")
class SignalProcessingTools:
    """Common preprocessing utilities for vibration signals."""

    def normalize(self, signal: np.ndarray) -> np.ndarray:
        """Standardize ``signal`` along the length dimension.

        Parameters
        ----------
        signal:
            Array of shape ``(B, L, C)`` or ``(B, L)`` or ``(L,)``.

        Returns
        -------
        np.ndarray
            Normalized signal with shape ``(B, L, C)``.
        """
        x = _ensure_3d(signal)
        mean = np.mean(x, axis=1, keepdims=True)
        std = np.std(x, axis=1, keepdims=True) + 1e-9
        return (x - mean) / std

    def detrend(self, signal: np.ndarray) -> np.ndarray:
        """Remove linear trend along the length dimension."""
        x = _ensure_3d(signal)
        detrended = scipy.signal.detrend(x, axis=1)
        return detrended

    def bandpass(
        self, signal: np.ndarray, fs: float, low: float, high: float, order: int = 4
    ) -> np.ndarray:
        """Apply a Butterworth bandpass filter.

        Parameters
        ----------
        signal:
            Input array ``(B, L, C)`` or lower-dimensional.
        fs:
            Sampling frequency.
        low:
            Low cutoff frequency in Hz.
        high:
            High cutoff frequency in Hz.
        order:
            Order of the Butterworth filter (default 4).
        """
        x = _ensure_3d(signal)
        b, a = scipy.signal.butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
        filtered = scipy.signal.filtfilt(b, a, x, axis=1)
        return filtered

    def fft(self, signal: np.ndarray) -> np.ndarray:
        """Return magnitude FFT of ``signal``.

        Parameters
        ----------
        signal:
            Array ``(B, L, C)`` or smaller.

        Returns
        -------
        np.ndarray
            Spectrum ``(B, F, C)`` where ``F`` is ``L/2+1``.
        """
        x = _ensure_3d(signal)
        spec = np.fft.rfft(x, axis=1)
        return np.abs(spec)

    def cepstrum(self, signal: np.ndarray) -> np.ndarray:
        """Compute real cepstrum of ``signal``."""
        spectrum = self.fft(signal)
        log_mag = np.log(spectrum + 1e-9)
        return np.fft.irfft(log_mag, axis=1)

    def envelope_spectrum(self, signal: np.ndarray) -> np.ndarray:
        """Hilbert envelope followed by FFT."""
        x = _ensure_3d(signal)
        analytic = scipy.signal.hilbert(x, axis=1)
        env = np.abs(analytic)
        return self.fft(env)

    def spectrogram(
        self,
        signal: np.ndarray,
        fs: float,
        nperseg: int = 256,
        noverlap: int | None = None,
    ) -> np.ndarray:
        """Compute STFT magnitude spectrogram.

        Returns array of shape ``(B, F, T, C)``.
        """
        x = _ensure_3d(signal)
        x = np.transpose(x, (0, 2, 1))  # B, C, L
        f, t, Zxx = scipy.signal.stft(
            x,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap if noverlap is not None else nperseg // 2,
            axis=2,
        )
        mag = np.abs(Zxx)
        return np.transpose(mag, (0, 2, 3, 1))

    def _mel_filterbank(
        self, n_fft: int, n_mels: int, fs: float, fmin: float, fmax: float
    ) -> np.ndarray:
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

    def mel_spectrogram(
        self,
        signal: np.ndarray,
        fs: float,
        nperseg: int = 256,
        noverlap: int | None = None,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: float | None = None,
    ) -> np.ndarray:
        """Compute mel-scale spectrogram.

        Returns array ``(B, M, T, C)`` with ``M`` mel bands.
        """
        spec = self.spectrogram(
            signal,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
        )
        power = spec**2
        if fmax is None:
            fmax = fs / 2
        fb = self._mel_filterbank(nperseg, n_mels, fs, fmin, fmax)
        mel = np.einsum("bftc,mf->bmtc", power, fb)
        return mel

    def scalogram(
        self,
        signal: np.ndarray,
        scales: np.ndarray | None = None,
        wavelet_width: float = 5.0,
    ) -> np.ndarray:
        """Continuous wavelet transform magnitude."""
        x = _ensure_3d(signal)
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

    def gramian_angular_field(self, signal: np.ndarray) -> np.ndarray:
        """Compute Gramian Angular Field for each signal."""
        x = _ensure_3d(signal)
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

    def markov_transition_field(
        self, signal: np.ndarray, bins: int = 8
    ) -> np.ndarray:
        """Compute Markov Transition Field."""
        x = _ensure_3d(signal)
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
                trans_sum = trans.sum(axis=1, keepdims=True) + 1e-9
                trans /= trans_sum
                for i in range(L):
                    for j in range(L):
                        out[b, i, j, c] = trans[states[i], states[j]]
        return out

    def recurrence_plot(self, signal: np.ndarray, eps: float | None = None) -> np.ndarray:
        """Binary recurrence plot based on pointwise distances."""
        x = _ensure_3d(signal)
        B, L, C = x.shape
        out = np.empty((B, L, L, C))
        for b in range(B):
            for c in range(C):
                s = x[b, :, c]
                if eps is None:
                    threshold = 0.1 * np.std(s)
                else:
                    threshold = eps
                dist = np.abs(s[:, None] - s[None, :])
                out[b, :, :, c] = (dist <= threshold).astype(float)
        return out

    def cepstrogram(
        self, signal: np.ndarray, fs: float, nperseg: int = 256, noverlap: int | None = None
    ) -> np.ndarray:
        """Cepstrum over time via STFT."""
        spec = self.spectrogram(signal, fs, nperseg, noverlap)
        log_spec = np.log(spec + 1e-9)
        ceps = np.fft.irfft(log_spec, axis=1)
        return ceps

    def envelope_spectrogram(
        self, signal: np.ndarray, fs: float, nperseg: int = 256, noverlap: int | None = None
    ) -> np.ndarray:
        """Spectrogram of the Hilbert envelope."""
        x = _ensure_3d(signal)
        analytic = scipy.signal.hilbert(x, axis=1)
        env = np.abs(analytic)
        return self.spectrogram(env, fs, nperseg, noverlap)


# ---------------------------------------------------------------------------
# Tool wrappers
# ---------------------------------------------------------------------------


@register_tool("normalize")
@tool
def normalize(signal: list | np.ndarray) -> np.ndarray:
    """Normalize a signal along its length dimension.
    
    Args:
        signal: The input signal to normalize.
    """
    return SignalProcessingTools().normalize(np.asarray(signal))


@register_tool("detrend")
@tool
def detrend(signal: list | np.ndarray) -> np.ndarray:
    """Remove linear trend from a signal.
    
    Args:
        signal: The input signal to detrend.
    """
    return SignalProcessingTools().detrend(np.asarray(signal))


@register_tool("bandpass")
@tool
def bandpass(signal: list | np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Bandpass filter a signal with a Butterworth design.
    
    Args:
        signal: The input signal to filter.
        fs: Sampling frequency.
        low: Low cutoff frequency.
        high: High cutoff frequency.
        order: Filter order.
    """
    return SignalProcessingTools().bandpass(np.asarray(signal), fs=fs, low=low, high=high, order=order)


@register_tool("fft")
@tool
def fft(signal: list | np.ndarray) -> np.ndarray:
    """Magnitude FFT of ``signal``.
    
    Args:
        signal: The input signal for FFT computation.
    """
    return SignalProcessingTools().fft(np.asarray(signal))


@register_tool("cepstrum")
@tool
def cepstrum(signal: list | np.ndarray) -> np.ndarray:
    """Real cepstrum of ``signal``.
    
    Args:
        signal: The input signal for cepstrum computation.
    """
    return SignalProcessingTools().cepstrum(np.asarray(signal))


@register_tool("envelope_spectrum")
@tool
def envelope_spectrum(signal: list | np.ndarray) -> np.ndarray:
    """Envelope spectrum via Hilbert transform.
    
    Args:
        signal: The input signal for envelope spectrum computation.
    """
    return SignalProcessingTools().envelope_spectrum(np.asarray(signal))


@register_tool("spectrogram")
@tool
def spectrogram(signal: list | np.ndarray, fs: float, nperseg: int = 256, noverlap: int | None = None) -> np.ndarray:
    """STFT magnitude spectrogram.
    
    Args:
        signal: The input signal for spectrogram computation.
        fs: Sampling frequency.
        nperseg: Length of each segment.
        noverlap: Number of points to overlap between segments.
    """
    return SignalProcessingTools().spectrogram(np.asarray(signal), fs=fs, nperseg=nperseg, noverlap=noverlap)


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
    """Mel scale spectrogram.
    
    Args:
        signal: The input signal for mel spectrogram computation.
        fs: Sampling frequency.
        nperseg: Length of each segment.
        noverlap: Number of points to overlap between segments.
        n_mels: Number of mel bands.
        fmin: Lowest frequency.
        fmax: Highest frequency.
    """
    return SignalProcessingTools().mel_spectrogram(
        np.asarray(signal),
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )


@register_tool("scalogram")
@tool
def scalogram(signal: list | np.ndarray, scales: list | None = None, wavelet_width: float = 5.0) -> np.ndarray:
    """Continuous wavelet transform magnitude.
    
    Args:
        signal: The input signal for scalogram computation.
        scales: The scales to use for the wavelet transform.
        wavelet_width: Width of the wavelet.
    """
    scales_arr = np.asarray(scales) if scales is not None else None
    return SignalProcessingTools().scalogram(np.asarray(signal), scales=scales_arr, wavelet_width=wavelet_width)


@register_tool("gramian_angular_field")
@tool
def gramian_angular_field(signal: list | np.ndarray) -> np.ndarray:
    """Gramian Angular Field.
    
    Args:
        signal: The input signal for Gramian Angular Field computation.
    """
    return SignalProcessingTools().gramian_angular_field(np.asarray(signal))


@register_tool("markov_transition_field")
@tool
def markov_transition_field(signal: list | np.ndarray, bins: int = 8) -> np.ndarray:
    """Markov Transition Field.
    
    Args:
        signal: The input signal for Markov Transition Field computation.
        bins: Number of bins for quantization.
    """
    return SignalProcessingTools().markov_transition_field(np.asarray(signal), bins=bins)


@register_tool("recurrence_plot")
@tool
def recurrence_plot(signal: list | np.ndarray, eps: float | None = None) -> np.ndarray:
    """Recurrence plot of a signal.
    
    Args:
        signal: The input signal for recurrence plot computation.
        eps: Threshold for recurrence.
    """
    return SignalProcessingTools().recurrence_plot(np.asarray(signal), eps=eps)


@register_tool("cepstrogram")
@tool
def cepstrogram(signal: list | np.ndarray, fs: float, nperseg: int = 256, noverlap: int | None = None) -> np.ndarray:
    """Cepstrum over time.
    
    Args:
        signal: The input signal for cepstrogram computation.
        fs: Sampling frequency.
        nperseg: Length of each segment.
        noverlap: Number of points to overlap between segments.
    """
    return SignalProcessingTools().cepstrogram(np.asarray(signal), fs=fs, nperseg=nperseg, noverlap=noverlap)


@register_tool("envelope_spectrogram")
@tool
def envelope_spectrogram(signal: list | np.ndarray, fs: float, nperseg: int = 256, noverlap: int | None = None) -> np.ndarray:
    """Spectrogram of the Hilbert envelope.
    
    Args:
        signal: The input signal for envelope spectrogram computation.
        fs: Sampling frequency.
        nperseg: Length of each segment.
        noverlap: Number of points to overlap between segments.
    """
    return SignalProcessingTools().envelope_spectrogram(np.asarray(signal), fs=fs, nperseg=nperseg, noverlap=noverlap)

