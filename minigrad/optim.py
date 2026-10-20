"""Optimizers for the educational minigrad framework."""
from __future__ import annotations
from typing import Iterable
import numpy as np
from .tensor import Tensor


class Optimizer:
    def __init__(self, parameters: Iterable[Tensor]):
        self.parameters = list(parameters)
        if not self.parameters:
            raise ValueError("optimizer received no trainable parameters")

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, parameters, lr: float = 1e-2, momentum: float = 0.0):
        super().__init__(parameters)
        if lr <= 0 or not 0 <= momentum < 1:
            raise ValueError("lr must be > 0 and momentum must be in [0, 1)")
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.velocity = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        for p, velocity in zip(self.parameters, self.velocity):
            if p.grad is None:
                continue
            velocity *= self.momentum
            velocity += p.grad
            p.data -= self.lr * velocity


class Adam(Optimizer):
    def __init__(self, parameters, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8):
        super().__init__(parameters)
        beta1, beta2 = betas
        if lr <= 0 or not (0 <= beta1 < 1 and 0 <= beta2 < 1) or eps <= 0:
            raise ValueError("invalid Adam hyperparameters")
        self.lr = float(lr)
        self.beta1, self.beta2 = float(beta1), float(beta2)
        self.eps = float(eps)
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        self.t += 1
        for p, m, v in zip(self.parameters, self.m, self.v):
            if p.grad is None:
                continue
            m *= self.beta1
            m += (1.0 - self.beta1) * p.grad
            v *= self.beta2
            v += (1.0 - self.beta2) * (p.grad ** 2)
            m_hat = m / (1.0 - self.beta1 ** self.t)
            v_hat = v / (1.0 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
