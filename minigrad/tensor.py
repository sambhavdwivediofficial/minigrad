from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union
import numpy as np

Array = np.ndarray
AxisArg = Union[int, Sequence[int], None]


def _as_array(value) -> Array:
    if isinstance(value, Tensor):
        return value.data
    return np.asarray(value, dtype=np.float64)


def _unbroadcast(grad: Array, target_shape: tuple[int, ...]) -> Array:
    grad = np.asarray(grad, dtype=np.float64)
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(target_shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(target_shape)


def _normalize_axis(axis: AxisArg, ndim: int) -> Optional[tuple[int, ...]]:
    if axis is None:
        return None
    if isinstance(axis, int):
        axis = (axis,)
    return tuple(sorted(a % ndim for a in axis))


def _expand_reduced_grad(grad: Array, axis: Optional[tuple[int, ...]], ndim: int) -> Array:
    if axis is None:
        return grad.reshape((1,) * ndim)
    shape = list(grad.shape)
    for a in axis:
        shape.insert(a, 1)
    return grad.reshape(shape)


class Tensor:

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

    @property
    def T(self) -> "Tensor":
        return self.transpose()

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

    def sum(self, axis: AxisArg = None, keepdims: bool = False) -> "Tensor":
        norm_axis = _normalize_axis(axis, self.data.ndim)
        value = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(value, self.requires_grad, (self,), "sum")

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if not keepdims:
                    grad = _expand_reduced_grad(grad, norm_axis, self.data.ndim)
                self.grad += np.ones_like(self.data) * grad

        out._backward = _backward
        return out

    def mean(self, axis: AxisArg = None, keepdims: bool = False) -> "Tensor":
        if self.data.size == 0:
            raise ValueError("mean() is undefined for an empty tensor")
        total = self.sum(axis=axis, keepdims=keepdims)
        if axis is None:
            count = self.data.size
        else:
            norm_axis = _normalize_axis(axis, self.data.ndim)
            count = 1
            for a in norm_axis:
                count *= self.data.shape[a]
        return total / count

    def max(self, axis: AxisArg = None, keepdims: bool = False) -> "Tensor":
        value = self.data.max(axis=axis, keepdims=True)
        mask = (self.data == value).astype(np.float64)
        mask_sum = mask.sum(axis=axis, keepdims=True)
        out_value = value if keepdims else (
            value.reshape(()) if axis is None else np.squeeze(value, axis=axis)
        )
        out = Tensor(out_value, self.requires_grad, (self,), "max")

        def _backward():
            if self.requires_grad:
                grad = out.grad.reshape(value.shape)
                self.grad += (mask / mask_sum) * grad

        out._backward = _backward
        return out

    def argmax(self, axis: AxisArg = None):
        return self.data.argmax(axis=axis)

    def reshape(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        value = self.data.reshape(shape)
        out = Tensor(value, self.requires_grad, (self,), "reshape")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def transpose(self, *axes) -> "Tensor":
        if len(axes) == 0:
            axes_arg = None
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes_arg = tuple(axes[0])
        else:
            axes_arg = tuple(axes)
        value = np.transpose(self.data, axes_arg)
        out = Tensor(value, self.requires_grad, (self,), "transpose")

        def _backward():
            if self.requires_grad:
                inv_axes = None if axes_arg is None else tuple(np.argsort(axes_arg))
                self.grad += np.transpose(out.grad, inv_axes)

        out._backward = _backward
        return out

    def __getitem__(self, index) -> "Tensor":
        value = self.data[index]
        out = Tensor(value, self.requires_grad, (self,), "getitem")

        def _backward():
            if self.requires_grad:
                grad_full = np.zeros_like(self.data)
                np.add.at(grad_full, index, out.grad)
                self.grad += grad_full

        out._backward = _backward
        return out

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

    def exp(self) -> "Tensor":
        value = np.exp(self.data)
        out = Tensor(value, self.requires_grad, (self,), "exp")

        def _backward():
            if self.requires_grad:
                self.grad += value * out.grad

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        out = Tensor(np.log(self.data), self.requires_grad, (self,), "log")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def abs(self) -> "Tensor":
        out = Tensor(np.abs(self.data), self.requires_grad, (self,), "abs")

        def _backward():
            if self.requires_grad:
                self.grad += np.sign(self.data) * out.grad

        out._backward = _backward
        return out

    def log_softmax(self, axis: int = -1) -> "Tensor":
        shifted = self - self.max(axis=axis, keepdims=True)
        log_sum_exp = shifted.exp().sum(axis=axis, keepdims=True).log()
        return shifted - log_sum_exp

    def softmax(self, axis: int = -1) -> "Tensor":
        return self.log_softmax(axis=axis).exp()

    def __repr__(self) -> str:
        return f"Tensor(data={self.data!r}, requires_grad={self.requires_grad})"


def where(condition: Array, a: Tensor, b: Tensor) -> Tensor:
    mask = np.asarray(condition, dtype=np.float64)
    a_t = a if isinstance(a, Tensor) else Tensor(a)
    b_t = b if isinstance(b, Tensor) else Tensor(b)
    value = mask * a_t.data + (1.0 - mask) * b_t.data
    requires_grad = a_t.requires_grad or b_t.requires_grad
    out = Tensor(value, requires_grad, (a_t, b_t), "where")

    def _backward():
        if a_t.requires_grad:
            a_t.grad += _unbroadcast(mask * out.grad, a_t.data.shape)
        if b_t.requires_grad:
            b_t.grad += _unbroadcast((1.0 - mask) * out.grad, b_t.data.shape)

    out._backward = _backward
    return out


def concatenate(tensors: Sequence[Tensor], axis: int = 0) -> Tensor:
    tensors = list(tensors)
    if not tensors:
        raise ValueError("concatenate requires at least one tensor")
    arrays = [t.data for t in tensors]
    value = np.concatenate(arrays, axis=axis)
    requires_grad = any(t.requires_grad for t in tensors)
    out = Tensor(value, requires_grad, tuple(tensors), "concat")
    sizes = [a.shape[axis] for a in arrays]
    ndim = value.ndim

    def _backward():
        start = 0
        for t, size in zip(tensors, sizes):
            if t.requires_grad:
                sl = [slice(None)] * ndim
                sl[axis] = slice(start, start + size)
                t.grad += out.grad[tuple(sl)]
            start += size

    out._backward = _backward
    return out


def stack(tensors: Sequence[Tensor], axis: int = 0) -> Tensor:
    tensors = list(tensors)
    if not tensors:
        raise ValueError("stack requires at least one tensor")
    expanded = []
    for t in tensors:
        shape = list(t.data.shape)
        insert_at = axis if axis >= 0 else axis + len(shape) + 1
        shape.insert(insert_at, 1)
        expanded.append(t.reshape(*shape))
    return concatenate(expanded, axis=axis)
