from __future__ import annotations

import numpy as np
import pytest

from redacted_backend.retrieval.int8_embeddings import (
    dequantize_directions,
    quantize_directions,
)


def test_quantize_directions_writes_deterministic_signed_bytes() -> None:
    values = np.asarray(
        [[1.0, -1.0, 0.0], [0.5, 0.25, -0.5]],
        dtype=np.float32,
    )

    quantized = quantize_directions(values)

    assert quantized.dtype == np.int8
    assert quantized.flags.c_contiguous
    assert quantized.tolist() == [[127, -127, 0], [127, 64, -127]]
    assert quantized.nbytes == values.shape[0] * values.shape[1]


def test_dequantize_directions_returns_unit_float32_rows_with_low_loss() -> None:
    original = np.asarray(
        [
            [0.8, 0.6, 0.0, 0.0],
            [0.5, -0.5, 0.5, -0.5],
            [0.1, 0.2, 0.3, 0.92736185],
        ],
        dtype=np.float32,
    )
    original /= np.linalg.norm(original, axis=1, keepdims=True)

    reconstructed = dequantize_directions(quantize_directions(original))

    assert reconstructed.dtype == np.float32
    assert reconstructed.flags.c_contiguous
    assert np.allclose(np.linalg.norm(reconstructed, axis=1), 1.0, atol=1e-6)
    assert np.all(np.sum(original * reconstructed, axis=1) >= 0.9999)


@pytest.mark.parametrize(
    "values",
    [
        np.asarray([1.0, 0.0], dtype=np.float32),
        np.asarray([[0.0, 0.0]], dtype=np.float32),
        np.asarray([[np.nan, 1.0]], dtype=np.float32),
        np.asarray([[np.inf, 1.0]], dtype=np.float32),
    ],
)
def test_quantize_directions_rejects_invalid_rows(values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="two-dimensional|finite nonzero"):
        quantize_directions(values)


def test_dequantize_directions_rejects_non_int8_and_zero_rows() -> None:
    with pytest.raises(ValueError, match="signed int8"):
        dequantize_directions(np.asarray([[1, 2]], dtype=np.int16))
    with pytest.raises(ValueError, match="nonzero"):
        dequantize_directions(np.asarray([[0, 0]], dtype=np.int8))
