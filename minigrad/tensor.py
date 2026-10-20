"""A tiny NumPy-backed Tensor with reverse-mode automatic differentiation.

Educational implementation: CPU only, float64, 2-D matrix multiplication,
and intentionally limited API. Not a replacement for PyTorch/JAX.
"""
from __future__ import annotations

from typing import Iterable, Optional
import numpy as np

Array = np.ndarray


def _as_array(value) -> Array:
    if isinstance(value, Tensor):
        return value.data
    return np.asarray(value, dtype=np.float64)


def _unbroadcast(grad: Array, target_shape: tuple[int, ...]) -> Array:
    """Undo NumPy broadcasting when propagating a gradient to an input."""
    grad = np.asarray(grad, dtype=np.float64)
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(target_shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(target_shape)


class Tensor:
    """An ndarray plus a small dynamic computation graph and gradient buffer."""

    __array_priority__ = 1000

    def __init__(
        self,
        data,
        requires_grad: bool = False,
        _children: Iterable["Tensor"] = (),
        _op: str = "",
    ):
        self.data: Array = np.asarray(data, dtype=np.float64)
        self.requires_grad = bool(requires_grad)
        self.grad: Optional[Array] = (
            np.zeros_like(self.data) if self.requires_grad else None
        )
        self._prev = tuple(_children)
        self._op = _op
        self._backward = lambda: None

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def item(self) -> float:
        return float(self.data.item())

    def numpy(self) -> Array:
        return self.data.copy()

    def detach(self) -> "Tensor":
        return Tensor(self.data.copy(), requires_grad=False)

    def zero_grad(self) -> None:
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def backward(self, gradient=None) -> None:
        """Backpropagate gradients from this tensor through its computation graph.

        For scalar outputs, the initial gradient defaults to 1. For non-scalar
        outputs, an explicit gradient with the same shape is required.
        Gradients accumulate until ``zero_grad`` is called on parameters.
        """
        if gradient is None:
            if self.data.size != 1:
                raise ValueError("gradient must be supplied for non-scalar outputs")
            gradient = np.ones_like(self.data)
        else:
            gradient = np.asarray(gradient, dtype=np.float64)
            if gradient.shape != self.data.shape:
                raise ValueError(
                    f"gradient shape {gradient.shape} must match output shape {self.data.shape}"
                )
        if not self.requires_grad:
            raise RuntimeError("cannot call backward() on a tensor that does not require gradients")

        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(node: Tensor):
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node._prev:
                build(parent)
            topo.append(node)

        build(self)
        assert self.grad is not None
        self.grad += gradient
        for node in reversed(topo):
            node._backward()

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, self.requires_grad or other.requires_grad, (self, other), "+")

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        out = Tensor(-self.data, self.requires_grad, (self,), "neg")

        def _backward():
            if self.requires_grad:
                self.grad -= out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-other if isinstance(other, Tensor) else -Tensor(other))

    def __rsub__(self, other):
        return Tensor(other) - self

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, self.requires_grad or other.requires_grad, (self, other), "*")

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, self.requires_grad or other.requires_grad, (self, other), "/")

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad / other.data, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(-out.grad * self.data / (other.data ** 2), other.data.shape)

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        return Tensor(other) / self

    def __pow__(self, power: float):
        if not np.isscalar(power):
            raise TypeError("Tensor currently supports scalar exponents only")
        out = Tensor(self.data ** power, self.requires_grad, (self,), f"pow({power})")

        def _backward():
            if self.requires_grad:
                self.grad += (power * self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        if self.data.ndim != 2 or other.data.ndim != 2:
            raise ValueError("@ currently supports 2-D matrices only")
        out = Tensor(self.data @ other.data, self.requires_grad or other.requires_grad, (self, other), "matmul")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def sum(self) -> "Tensor":
        out = Tensor(np.array(self.data.sum()), self.requires_grad, (self,), "sum")

        def _backward():
            if self.requires_grad:
                self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out

    def mean(self) -> "Tensor":
        if self.data.size == 0:
            raise ValueError("mean() is undefined for an empty tensor")
        return self.sum() / self.data.size

    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(self.data, 0.0), self.requires_grad, (self,), "relu")

        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0.0) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        value = np.tanh(self.data)
        out = Tensor(value, self.requires_grad, (self,), "tanh")

        def _backward():
            if self.requires_grad:
                self.grad += (1.0 - value ** 2) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        # Numerically stable sigmoid implementation.
        value = np.empty_like(self.data)
        positive = self.data >= 0
        value[positive] = 1.0 / (1.0 + np.exp(-self.data[positive]))
        exp_x = np.exp(self.data[~positive])
        value[~positive] = exp_x / (1.0 + exp_x)
        out = Tensor(value, self.requires_grad, (self,), "sigmoid")

        def _backward():
            if self.requires_grad:
                self.grad += value * (1.0 - value) * out.grad

        out._backward = _backward
        return out

    def softplus(self) -> "Tensor":
        """Stable log(1 + exp(x)), useful for binary cross-entropy with logits."""
        value = np.logaddexp(0.0, self.data)
        out = Tensor(value, self.requires_grad, (self,), "softplus")

        def _backward():
            if self.requires_grad:
                sig = np.empty_like(self.data)
                positive = self.data >= 0
                sig[positive] = 1.0 / (1.0 + np.exp(-self.data[positive]))
                exp_x = np.exp(self.data[~positive])
                sig[~positive] = exp_x / (1.0 + exp_x)
                self.grad += sig * out.grad

        out._backward = _backward
        return out

    def __repr__(self) -> str:
        return f"Tensor(data={self.data!r}, requires_grad={self.requires_grad})"
