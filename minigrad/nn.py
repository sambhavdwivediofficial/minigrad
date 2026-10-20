"""Minimal neural-network layers and loss functions built on minigrad.Tensor."""
from __future__ import annotations

from typing import Iterable
import numpy as np
from .tensor import Tensor


class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def named_parameters(self, prefix: str = ""):
        """Yield (stable_name, trainable_tensor) pairs recursively."""
        seen: set[int] = set()

        def walk(obj, name):
            if isinstance(obj, Tensor):
                if obj.requires_grad and id(obj) not in seen:
                    seen.add(id(obj))
                    yield name, obj
            elif isinstance(obj, Module):
                for key, value in obj.__dict__.items():
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

    def state_dict(self):
        return {name: parameter.data.copy() for name, parameter in self.named_parameters()}

    def load_state_dict(self, state: dict, strict: bool = True):
        current = dict(self.named_parameters())
        if strict and set(current) != set(state):
            missing = sorted(set(current) - set(state))
            extra = sorted(set(state) - set(current))
            raise ValueError(f"checkpoint keys mismatch; missing={missing}, extra={extra}")
        for name, value in state.items():
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


class Linear(Module):
    """Fully connected layer: input [batch, in_features] -> [batch, out_features]."""
    def __init__(self, in_features: int, out_features: int, seed: int | None = None):
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        rng = np.random.default_rng(seed)
        # He-style scale works reasonably for tanh/relu educational examples;
        # larger projects should use activation-aware initialization.
        scale = np.sqrt(2.0 / in_features)
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


class Sequential(Module):
    def __init__(self, *layers: Module):
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def mse_loss(prediction: Tensor, target: Tensor) -> Tensor:
    return ((prediction - target) ** 2).mean()


def binary_cross_entropy_with_logits(logits: Tensor, target: Tensor) -> Tensor:
    """Stable BCE-with-logits; targets should contain values 0 or 1."""
    return (logits.softplus() - logits * target).mean()
