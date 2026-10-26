from __future__ import annotations

from typing import Callable, Sequence, Union
import numpy as np
from .tensor import Tensor


def grad(output: Tensor, inputs: Union[Tensor, Sequence[Tensor]], create_graph: bool = False):
    single = isinstance(inputs, Tensor)
    input_list = [inputs] if single else list(inputs)
    for t in input_list:
        if not t.requires_grad:
            raise ValueError("every tensor in `inputs` must have requires_grad=True")
        t.grad = None
    if output.data.size != 1:
        raise ValueError("grad() requires a scalar `output` Tensor")
    output.backward(create_graph=create_graph)
    results = [t.grad for t in input_list]
    return results[0] if single else results


def hvp(fn: Callable[[Tensor], Tensor], x: Tensor, v) -> np.ndarray:
    if x.data.ndim != 1:
        raise ValueError("hvp() currently supports a single 1-D Tensor input only")
    v = np.asarray(v, dtype=np.float64)
    if v.shape != x.data.shape:
        raise ValueError(f"v shape {v.shape} must match x shape {x.data.shape}")
    xt = Tensor(x.data.copy(), requires_grad=True)
    output = fn(xt)
    if output.data.size != 1:
        raise ValueError("fn(x) must return a scalar Tensor")
    output.backward(create_graph=True)
    g = xt.grad
    directional = (g * Tensor(v)).sum()
    xt.grad = None
    directional.backward()
    return xt.grad.copy()


def hessian(fn: Callable[[Tensor], Tensor], x: Tensor) -> np.ndarray:
    if x.data.ndim != 1:
        raise ValueError("hessian() currently supports a single 1-D Tensor input only")
    n = x.data.size
    rows = np.zeros((n, n))
    for i in range(n):
        v = np.zeros(n)
        v[i] = 1.0
        rows[i] = hvp(fn, x, v)
    return rows


def newton_step(fn: Callable[[Tensor], Tensor], x: Tensor, damping: float = 1e-6) -> np.ndarray:
    xt = Tensor(x.data.copy(), requires_grad=True)
    output = fn(xt)
    if output.data.size != 1:
        raise ValueError("fn(x) must return a scalar Tensor")
    output.backward(create_graph=True)
    g = xt.grad
    g_array = g.data.copy()
    hess = hessian(fn, x)
    hess = hess + damping * np.eye(hess.shape[0])
    step = np.linalg.solve(hess, g_array)
    return x.data - step
