from __future__ import annotations

from typing import Iterable
import numpy as np
from .tensor import Tensor, where


class Module:
    def __init__(self):
        self.training: bool = True
        self._buffers: dict[str, np.ndarray] = {}

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def register_buffer(self, name: str, value) -> None:
        if not hasattr(self, "_buffers"):
            self._buffers = {}
        self._buffers[name] = np.asarray(value, dtype=np.float64)

    def buffer(self, name: str) -> np.ndarray:
        return self._buffers[name]

    def set_buffer(self, name: str, value) -> None:
        self._buffers[name] = np.asarray(value, dtype=np.float64)

    def train(self, mode: bool = True) -> "Module":
        self.training = mode
        for value in self.__dict__.values():
            Module._set_mode(value, mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    @staticmethod
    def _set_mode(obj, mode: bool) -> None:
        if isinstance(obj, Module):
            obj.training = mode
            for value in obj.__dict__.values():
                Module._set_mode(value, mode)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                Module._set_mode(value, mode)
        elif isinstance(obj, dict):
            for value in obj.values():
                Module._set_mode(value, mode)

    def named_parameters(self, prefix: str = ""):
        seen: set[int] = set()

        def walk(obj, name):
            if isinstance(obj, Tensor):
                if obj.requires_grad and id(obj) not in seen:
                    seen.add(id(obj))
                    yield name, obj
            elif isinstance(obj, Module):
                for key, value in obj.__dict__.items():
                    if key == "_buffers":
                        continue
                    child_name = f"{name}.{key}" if name else key
                    yield from walk(value, child_name)
            elif isinstance(obj, (list, tuple)):
                for idx, value in enumerate(obj):
                    child_name = f"{name}.{idx}" if name else str(idx)
                    yield from walk(value, child_name)
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    child_name = f"{name}.{key}" if name else str(key)
                    yield from walk(value, child_name)

        yield from walk(self, prefix)

    def parameters(self):
        for _, parameter in self.named_parameters():
            yield parameter

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def num_parameters(self) -> int:
        return int(sum(p.data.size for p in self.parameters()))

    def named_buffers(self, prefix: str = ""):
        def walk(obj, name):
            if isinstance(obj, Module):
                for key, value in obj._buffers.items():
                    yield (f"{name}.{key}" if name else key), value
                for key, value in obj.__dict__.items():
                    if key == "_buffers":
                        continue
                    child_name = f"{name}.{key}" if name else key
                    yield from walk(value, child_name)
            elif isinstance(obj, (list, tuple)):
                for idx, value in enumerate(obj):
                    yield from walk(value, f"{name}.{idx}" if name else str(idx))
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    yield from walk(value, f"{name}.{key}" if name else str(key))

        yield from walk(self, prefix)

    def load_buffers(self, buffers: dict) -> None:
        remaining = dict(buffers)

        def walk(obj, name):
            if isinstance(obj, Module):
                for key in list(obj._buffers.keys()):
                    full = f"{name}.{key}" if name else key
                    if full in remaining:
                        obj._buffers[key] = np.asarray(remaining.pop(full), dtype=np.float64)
                for key, value in obj.__dict__.items():
                    if key == "_buffers":
                        continue
                    child_name = f"{name}.{key}" if name else key
                    walk(value, child_name)
            elif isinstance(obj, (list, tuple)):
                for idx, value in enumerate(obj):
                    walk(value, f"{name}.{idx}" if name else str(idx))
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    walk(value, f"{name}.{key}" if name else str(key))

        walk(self, "")

    def state_dict(self) -> dict:
        state = {name: parameter.data.copy() for name, parameter in self.named_parameters()}
        for name, buf in self.named_buffers():
            state[f"__buffer__.{name}"] = buf.copy()
        return state

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        params_in_state = {k: v for k, v in state.items() if not k.startswith("__buffer__.")}
        buffers_in_state = {
            k[len("__buffer__."):]: v for k, v in state.items() if k.startswith("__buffer__.")
        }

        current = dict(self.named_parameters())
        if strict and set(current) != set(params_in_state):
            missing = sorted(set(current) - set(params_in_state))
            extra = sorted(set(params_in_state) - set(current))
            raise ValueError(f"checkpoint keys mismatch; missing={missing}, extra={extra}")
        for name, value in params_in_state.items():
            if name not in current:
                if strict:
                    raise ValueError(f"unexpected parameter: {name}")
                continue
            value = np.asarray(value, dtype=np.float64)
            if current[name].data.shape != value.shape:
                raise ValueError(
                    f"shape mismatch for {name}: model {current[name].data.shape}, checkpoint {value.shape}"
                )
            current[name].data = value.copy()

        if buffers_in_state:
            self.load_buffers(buffers_in_state)


class Linear(Module):

    def __init__(self, in_features: int, out_features: int, seed: int | None = None, init: str = "he"):
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        rng = np.random.default_rng(seed)
        if init == "he":
            scale = np.sqrt(2.0 / in_features)
        elif init == "xavier":
            scale = np.sqrt(1.0 / in_features)
        elif init == "zeros":
            scale = 0.0
        else:
            raise ValueError(f"unknown init scheme: {init!r} (use 'he', 'xavier', or 'zeros')")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(rng.normal(0.0, scale, (in_features, out_features)), requires_grad=True)
        self.bias = Tensor(np.zeros((1, out_features)), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Flatten(Module):

    def forward(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        size = 1
        for d in x.shape[1:]:
            size *= d
        return x.reshape(batch, size)


class Dropout(Module):

    def __init__(self, p: float = 0.5, seed: int | None = None):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout probability p must be in [0, 1)")
        self.p = float(p)
        self._rng = np.random.default_rng(seed)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        mask = (self._rng.random(x.data.shape) < keep_prob).astype(np.float64) / keep_prob
        return x * Tensor(mask)


class LayerNorm(Module):

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)
        self.gamma = Tensor(np.ones((1, normalized_shape)), requires_grad=True)
        self.beta = Tensor(np.zeros((1, normalized_shape)), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(axis=-1, keepdims=True)
        centered = x - mean
        var = (centered * centered).mean(axis=-1, keepdims=True)
        normalized = centered / (var + self.eps) ** 0.5
        return normalized * self.gamma + self.beta


class BatchNorm1d(Module):

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.gamma = Tensor(np.ones((1, num_features)), requires_grad=True)
        self.beta = Tensor(np.zeros((1, num_features)), requires_grad=True)
        self.register_buffer("running_mean", np.zeros((1, num_features)))
        self.register_buffer("running_var", np.ones((1, num_features)))

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            batch_mean = x.mean(axis=0, keepdims=True)
            centered = x - batch_mean
            batch_var = (centered * centered).mean(axis=0, keepdims=True)
            running_mean = self.buffer("running_mean")
            running_var = self.buffer("running_var")
            self.set_buffer("running_mean", (1 - self.momentum) * running_mean + self.momentum * batch_mean.data)
            self.set_buffer("running_var", (1 - self.momentum) * running_var + self.momentum * batch_var.data)
            normalized = centered / (batch_var + self.eps) ** 0.5
        else:
            running_mean = Tensor(self.buffer("running_mean"))
            running_var = Tensor(self.buffer("running_var"))
            normalized = (x - running_mean) / (running_var + self.eps) ** 0.5
        return normalized * self.gamma + self.beta


class Sequential(Module):
    def __init__(self, *layers: Module):
        super().__init__()
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def mse_loss(prediction: Tensor, target: Tensor) -> Tensor:
    return ((prediction - target) ** 2).mean()


def mae_loss(prediction: Tensor, target: Tensor) -> Tensor:
    return (prediction - target).abs().mean()


def huber_loss(prediction: Tensor, target: Tensor, delta: float = 1.0) -> Tensor:
    diff = prediction - target
    abs_diff = diff.abs()
    quadratic = (diff * diff) * 0.5
    linear = (abs_diff - 0.5 * delta) * delta
    elementwise = where(abs_diff.data <= delta, quadratic, linear)
    return elementwise.mean()


def binary_cross_entropy_with_logits(logits: Tensor, target: Tensor) -> Tensor:
    return (logits.softplus() - logits * target).mean()


def softmax_cross_entropy(logits: Tensor, targets) -> Tensor:
    targets = np.asarray(targets)
    if targets.ndim != 1:
        raise ValueError("targets must be a 1-D array of integer class indices")
    if logits.ndim != 2 or targets.shape[0] != logits.shape[0]:
        raise ValueError("logits must be [batch, num_classes] and match targets' batch size")
    batch = logits.shape[0]
    log_probs = logits.log_softmax(axis=1)
    picked = log_probs[(np.arange(batch), targets)]
    return -picked.mean()


def kl_divergence(log_probs: Tensor, target_probs: Tensor) -> Tensor:
    target_data = target_probs.data if isinstance(target_probs, Tensor) else np.asarray(target_probs)
    safe_target = np.clip(target_data, 1e-12, 1.0)
    target_t = Tensor(target_data)
    entropy_term = float(np.sum(safe_target * np.log(safe_target), axis=-1).mean())
    cross_term = (target_t * log_probs).sum(axis=-1).mean()
    return Tensor(entropy_term) - cross_term
