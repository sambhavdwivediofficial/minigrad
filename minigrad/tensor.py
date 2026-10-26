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


def _unbroadcast_t(grad: "Tensor", target_shape: tuple[int, ...]) -> "Tensor":
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(target_shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(target_shape)


def _unbroadcast_matmul(grad: Array, target_shape: tuple[int, ...]) -> Array:
    while grad.ndim - 2 > len(target_shape) - 2:
        grad = grad.sum(axis=0)
    target_batch = target_shape[:-2]
    for axis, size in enumerate(target_batch):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(target_shape)


def _unbroadcast_matmul_t(grad: "Tensor", target_shape: tuple[int, ...]) -> "Tensor":
    while grad.ndim - 2 > len(target_shape) - 2:
        grad = grad.sum(axis=0)
    target_batch = target_shape[:-2]
    for axis, size in enumerate(target_batch):
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


def _expand_reduced_grad_t(grad: "Tensor", axis: Optional[tuple[int, ...]], ndim: int) -> "Tensor":
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
        self.grad = (
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

    def _accumulate_grad(self, contribution) -> None:
        if isinstance(contribution, Tensor):
            if isinstance(self.grad, Tensor):
                self.grad = self.grad + contribution
            else:
                baseline_data = self.grad if self.grad is not None else np.zeros_like(self.data)
                self.grad = Tensor(baseline_data.copy()) + contribution
        else:
            if self.grad is None or isinstance(self.grad, Tensor):
                self.grad = np.zeros_like(self.data)
            self.grad += contribution

    def backward(self, gradient=None, create_graph: bool = False) -> None:
        if gradient is None:
            if self.data.size != 1:
                raise ValueError("gradient must be supplied for non-scalar outputs")
            gradient = Tensor(np.ones_like(self.data)) if create_graph else np.ones_like(self.data)
        else:
            if create_graph:
                if not isinstance(gradient, Tensor):
                    gradient = Tensor(np.asarray(gradient, dtype=np.float64))
                if gradient.shape != self.data.shape:
                    raise ValueError(
                        f"gradient shape {gradient.shape} must match output shape {self.data.shape}"
                    )
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
        self._accumulate_grad(gradient)
        for node in reversed(topo):
            node._backward()

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, self.requires_grad or other.requires_grad, (self, other), "+")

        def _backward():
            tensor_mode = isinstance(out.grad, Tensor)
            if self.requires_grad:
                self._accumulate_grad(
                    _unbroadcast_t(out.grad, self.data.shape) if tensor_mode
                    else _unbroadcast(out.grad, self.data.shape)
                )
            if other.requires_grad:
                other._accumulate_grad(
                    _unbroadcast_t(out.grad, other.data.shape) if tensor_mode
                    else _unbroadcast(out.grad, other.data.shape)
                )

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        out = Tensor(-self.data, self.requires_grad, (self,), "neg")

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(-out.grad)

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
            tensor_mode = isinstance(out.grad, Tensor)
            if self.requires_grad:
                raw = (other * out.grad) if tensor_mode else (other.data * out.grad)
                self._accumulate_grad(
                    _unbroadcast_t(raw, self.data.shape) if tensor_mode else _unbroadcast(raw, self.data.shape)
                )
            if other.requires_grad:
                raw = (self * out.grad) if tensor_mode else (self.data * out.grad)
                other._accumulate_grad(
                    _unbroadcast_t(raw, other.data.shape) if tensor_mode else _unbroadcast(raw, other.data.shape)
                )

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, self.requires_grad or other.requires_grad, (self, other), "/")

        def _backward():
            tensor_mode = isinstance(out.grad, Tensor)
            if self.requires_grad:
                raw = (out.grad / other) if tensor_mode else (out.grad / other.data)
                self._accumulate_grad(
                    _unbroadcast_t(raw, self.data.shape) if tensor_mode else _unbroadcast(raw, self.data.shape)
                )
            if other.requires_grad:
                raw = (-out.grad * self / (other ** 2)) if tensor_mode else (
                    -out.grad * self.data / (other.data ** 2)
                )
                other._accumulate_grad(
                    _unbroadcast_t(raw, other.data.shape) if tensor_mode else _unbroadcast(raw, other.data.shape)
                )

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
                if isinstance(out.grad, Tensor):
                    self._accumulate_grad((power * self ** (power - 1)) * out.grad)
                else:
                    self._accumulate_grad((power * self.data ** (power - 1)) * out.grad)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        if self.data.ndim < 2 or other.data.ndim < 2:
            raise ValueError("@ requires both operands to have at least 2 dimensions")
        if self.data.shape[-1] != other.data.shape[-2]:
            raise ValueError(f"matmul shape mismatch: {self.data.shape} @ {other.data.shape}")
        out = Tensor(np.matmul(self.data, other.data), self.requires_grad or other.requires_grad, (self, other), "matmul")

        def _backward():
            tensor_mode = isinstance(out.grad, Tensor)
            if self.requires_grad:
                if tensor_mode:
                    grad_a = out.grad @ other.swapaxes(-1, -2)
                    self._accumulate_grad(_unbroadcast_matmul_t(grad_a, self.data.shape))
                else:
                    grad_a = np.matmul(out.grad, np.swapaxes(other.data, -1, -2))
                    self._accumulate_grad(_unbroadcast_matmul(grad_a, self.data.shape))
            if other.requires_grad:
                if tensor_mode:
                    grad_b = self.swapaxes(-1, -2) @ out.grad
                    other._accumulate_grad(_unbroadcast_matmul_t(grad_b, other.data.shape))
                else:
                    grad_b = np.matmul(np.swapaxes(self.data, -1, -2), out.grad)
                    other._accumulate_grad(_unbroadcast_matmul(grad_b, other.data.shape))

        out._backward = _backward
        return out

    def sum(self, axis: AxisArg = None, keepdims: bool = False) -> "Tensor":
        norm_axis = _normalize_axis(axis, self.data.ndim)
        value = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(value, self.requires_grad, (self,), "sum")

        def _backward():
            if self.requires_grad:
                if isinstance(out.grad, Tensor):
                    grad = out.grad if keepdims else _expand_reduced_grad_t(out.grad, norm_axis, self.data.ndim)
                    self._accumulate_grad(Tensor(np.zeros(self.data.shape)) + grad)
                else:
                    grad = out.grad if keepdims else _expand_reduced_grad(out.grad, norm_axis, self.data.ndim)
                    self._accumulate_grad(np.ones_like(self.data) * grad)

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
                if isinstance(out.grad, Tensor):
                    self._accumulate_grad(Tensor(mask / mask_sum) * grad)
                else:
                    self._accumulate_grad((mask / mask_sum) * grad)

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
                self._accumulate_grad(out.grad.reshape(self.data.shape))

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
                if isinstance(out.grad, Tensor):
                    self._accumulate_grad(out.grad.transpose(*inv_axes) if inv_axes is not None else out.grad.transpose())
                else:
                    self._accumulate_grad(np.transpose(out.grad, inv_axes))

        out._backward = _backward
        return out

    def swapaxes(self, axis1: int, axis2: int) -> "Tensor":
        ndim = self.data.ndim
        axis1 = axis1 % ndim
        axis2 = axis2 % ndim
        perm = list(range(ndim))
        perm[axis1], perm[axis2] = perm[axis2], perm[axis1]
        return self.transpose(*perm)

    def __getitem__(self, index) -> "Tensor":
        value = self.data[index]
        out = Tensor(value, self.requires_grad, (self,), "getitem")

        def _backward():
            if self.requires_grad:
                if isinstance(out.grad, Tensor):
                    raise NotImplementedError(
                        "second-order gradients are not yet supported through indexing/slicing "
                        "(Tensor.__getitem__). This blocks create_graph=True for any forward pass "
                        "that uses fancy or slice indexing internally: Conv2d, MaxPool2d, AvgPool2d, "
                        "Embedding, softmax_cross_entropy, and the RNN/LSTM/GRU sequence wrappers all "
                        "do. Plain stacks of +, -, *, /, **, @ (matmul), sum, mean, max, reshape, "
                        "transpose, relu, tanh, sigmoid, softplus, exp, log, abs, where, concatenate "
                        "and stack (so: Linear, LayerNorm, BatchNorm1d, Dropout, mse/mae/huber/"
                        "binary_cross_entropy_with_logits loss, and a single RNNCell/LSTMCell/GRUCell "
                        "call) fully support create_graph=True. See LIMITATIONS.md."
                    )
                grad_full = np.zeros_like(self.data)
                np.add.at(grad_full, index, out.grad)
                self._accumulate_grad(grad_full)

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(self.data, 0.0), self.requires_grad, (self,), "relu")

        def _backward():
            if self.requires_grad:
                mask = self.data > 0.0
                if isinstance(out.grad, Tensor):
                    self._accumulate_grad(Tensor(mask.astype(np.float64)) * out.grad)
                else:
                    self._accumulate_grad(mask * out.grad)

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        value = np.tanh(self.data)
        out = Tensor(value, self.requires_grad, (self,), "tanh")

        def _backward():
            if self.requires_grad:
                if isinstance(out.grad, Tensor):
                    self._accumulate_grad((Tensor(1.0) - self.tanh() ** 2) * out.grad)
                else:
                    self._accumulate_grad((1.0 - value ** 2) * out.grad)

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
                if isinstance(out.grad, Tensor):
                    s = self.sigmoid()
                    self._accumulate_grad(s * (Tensor(1.0) - s) * out.grad)
                else:
                    self._accumulate_grad(value * (1.0 - value) * out.grad)

        out._backward = _backward
        return out

    def softplus(self) -> "Tensor":
        value = np.logaddexp(0.0, self.data)
        out = Tensor(value, self.requires_grad, (self,), "softplus")

        def _backward():
            if self.requires_grad:
                if isinstance(out.grad, Tensor):
                    self._accumulate_grad(self.sigmoid() * out.grad)
                else:
                    sig = np.empty_like(self.data)
                    positive = self.data >= 0
                    sig[positive] = 1.0 / (1.0 + np.exp(-self.data[positive]))
                    exp_x = np.exp(self.data[~positive])
                    sig[~positive] = exp_x / (1.0 + exp_x)
                    self._accumulate_grad(sig * out.grad)

        out._backward = _backward
        return out

    def exp(self) -> "Tensor":
        value = np.exp(self.data)
        out = Tensor(value, self.requires_grad, (self,), "exp")

        def _backward():
            if self.requires_grad:
                if isinstance(out.grad, Tensor):
                    self._accumulate_grad(self.exp() * out.grad)
                else:
                    self._accumulate_grad(value * out.grad)

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        out = Tensor(np.log(self.data), self.requires_grad, (self,), "log")

        def _backward():
            if self.requires_grad:
                if isinstance(out.grad, Tensor):
                    self._accumulate_grad(out.grad / self)
                else:
                    self._accumulate_grad(out.grad / self.data)

        out._backward = _backward
        return out

    def abs(self) -> "Tensor":
        out = Tensor(np.abs(self.data), self.requires_grad, (self,), "abs")

        def _backward():
            if self.requires_grad:
                sign = np.sign(self.data)
                if isinstance(out.grad, Tensor):
                    self._accumulate_grad(Tensor(sign) * out.grad)
                else:
                    self._accumulate_grad(sign * out.grad)

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
        tensor_mode = isinstance(out.grad, Tensor)
        if a_t.requires_grad:
            raw = (Tensor(mask) * out.grad) if tensor_mode else (mask * out.grad)
            a_t._accumulate_grad(
                _unbroadcast_t(raw, a_t.data.shape) if tensor_mode else _unbroadcast(raw, a_t.data.shape)
            )
        if b_t.requires_grad:
            raw = (Tensor(1.0 - mask) * out.grad) if tensor_mode else ((1.0 - mask) * out.grad)
            b_t._accumulate_grad(
                _unbroadcast_t(raw, b_t.data.shape) if tensor_mode else _unbroadcast(raw, b_t.data.shape)
            )

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
                t._accumulate_grad(out.grad[tuple(sl)])
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
