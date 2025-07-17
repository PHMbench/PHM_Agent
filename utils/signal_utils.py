from __future__ import annotations

"""Utility helpers for signal processing modules."""

import numpy as np


def ensure_3d(x: np.ndarray) -> np.ndarray:
    """Ensure input ``x`` has shape ``(B, L, C)``.

    1D arrays become ``(1, L, 1)`` and 2D arrays become ``(B, L, 1)``.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        return x[None, :, None]
    if x.ndim == 2:
        return x[:, :, None]
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected 1D, 2D, or 3D array, got shape {x.shape}")
