"""Numerical utilities for checking custom automatic-differentiation code."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .tensor import Tensor


@dataclass(frozen=True)
class GradcheckResult:
    passed: bool
    max_absolute_error: float
    max_relative_error: float


def gradcheck(fn, x: Tensor, eps: float = 1e-6, atol: float = 1e-5, rtol: float = 1e-3) -> GradcheckResult:
    """Compare autodiff gradients to central finite differences for scalar fn(x).

    This simple checker handles one Tensor input and a scalar Tensor output.
    It is intentionally slow and intended for unit tests, not training.
    """
    if not x.requires_grad:
        raise ValueError("gradcheck input must have requires_grad=True")
    if eps <= 0:
        raise ValueError("eps must be positive")

    original = x.data.copy()
    x.zero_grad()
    output = fn(x)
    if not isinstance(output, Tensor) or output.data.size != 1:
        raise ValueError("fn(x) must return a scalar Tensor")
    output.backward()
    analytical = x.grad.copy()

    numerical = np.zeros_like(original)
    try:
        for index in np.ndindex(original.shape):
            x.data[index] = original[index] + eps
            plus = float(fn(x).data.item())
            x.data[index] = original[index] - eps
            minus = float(fn(x).data.item())
            numerical[index] = (plus - minus) / (2.0 * eps)
    finally:
        x.data[...] = original

    absolute = np.abs(analytical - numerical)
    denominator = np.maximum(np.abs(analytical) + np.abs(numerical), 1e-12)
    relative = absolute / denominator
    passed = bool(np.allclose(analytical, numerical, atol=atol, rtol=rtol))
    return GradcheckResult(
        passed=passed,
        max_absolute_error=float(absolute.max(initial=0.0)),
        max_relative_error=float(relative.max(initial=0.0)),
    )
