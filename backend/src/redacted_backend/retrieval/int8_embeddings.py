from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def quantize_directions(values: NDArray[np.floating]) -> NDArray[np.int8]:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Dense directions must be a two-dimensional matrix")
    maximum = np.max(np.abs(matrix), axis=1, keepdims=True)
    if np.any(~np.isfinite(matrix)) or np.any(maximum <= 0):
        raise ValueError("Dense directions must contain finite nonzero rows")
    scaled = np.rint(matrix / maximum * 127.0)
    return np.ascontiguousarray(np.clip(scaled, -127, 127), dtype=np.int8)


def dequantize_directions(values: NDArray[np.integer]) -> NDArray[np.float32]:
    quantized = np.asarray(values)
    if quantized.ndim != 2:
        raise ValueError("Dense int8 directions must be a two-dimensional matrix")
    if quantized.dtype != np.int8:
        raise ValueError("Dense directions must use signed int8 storage")
    matrix = np.ascontiguousarray(quantized, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("Dense int8 directions must contain nonzero rows")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)
