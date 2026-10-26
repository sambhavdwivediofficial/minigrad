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

    def state_dict(self) -> dict:
        raise NotImplementedError

    def load_state_dict(self, state: dict) -> None:
        raise NotImplementedError

    def _pack_buffers(self, state: dict, name: str, buffers) -> None:
        for i, buf in enumerate(buffers):
            state[f"{name}.{i}"] = buf.copy()

    def _unpack_buffers(self, state: dict, name: str, buffers) -> None:
        for i in range(len(buffers)):
            key = f"{name}.{i}"
            if key not in state:
                raise ValueError(f"checkpoint is missing optimizer buffer {key!r}")
            buffers[i] = np.asarray(state[key], dtype=np.float64).copy()


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

    def state_dict(self) -> dict:
        state = {"lr": np.array(self.lr), "momentum": np.array(self.momentum)}
        self._pack_buffers(state, "velocity", self.velocity)
        return state

    def load_state_dict(self, state: dict) -> None:
        self.lr = float(state["lr"])
        self.momentum = float(state["momentum"])
        self._unpack_buffers(state, "velocity", self.velocity)


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

    def state_dict(self) -> dict:
        state = {
            "lr": np.array(self.lr), "beta1": np.array(self.beta1), "beta2": np.array(self.beta2),
            "eps": np.array(self.eps), "t": np.array(self.t),
        }
        self._pack_buffers(state, "m", self.m)
        self._pack_buffers(state, "v", self.v)
        return state

    def load_state_dict(self, state: dict) -> None:
        self.lr = float(state["lr"])
        self.beta1 = float(state["beta1"])
        self.beta2 = float(state["beta2"])
        self.eps = float(state["eps"])
        self.t = int(state["t"])
        self._unpack_buffers(state, "m", self.m)
        self._unpack_buffers(state, "v", self.v)


class RMSProp(Optimizer):

    def __init__(self, parameters, lr: float = 1e-2, alpha: float = 0.99, eps: float = 1e-8, momentum: float = 0.0):
        super().__init__(parameters)
        if lr <= 0 or not 0 <= alpha < 1 or eps <= 0 or not 0 <= momentum < 1:
            raise ValueError("invalid RMSProp hyperparameters")
        self.lr = float(lr)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.square_avg = [np.zeros_like(p.data) for p in self.parameters]
        self.momentum_buffer = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        for p, sq, buf in zip(self.parameters, self.square_avg, self.momentum_buffer):
            if p.grad is None:
                continue
            sq *= self.alpha
            sq += (1.0 - self.alpha) * (p.grad ** 2)
            update = p.grad / (np.sqrt(sq) + self.eps)
            if self.momentum > 0:
                buf *= self.momentum
                buf += update
                p.data -= self.lr * buf
            else:
                p.data -= self.lr * update

    def state_dict(self) -> dict:
        state = {
            "lr": np.array(self.lr), "alpha": np.array(self.alpha),
            "eps": np.array(self.eps), "momentum": np.array(self.momentum),
        }
        self._pack_buffers(state, "square_avg", self.square_avg)
        self._pack_buffers(state, "momentum_buffer", self.momentum_buffer)
        return state

    def load_state_dict(self, state: dict) -> None:
        self.lr = float(state["lr"])
        self.alpha = float(state["alpha"])
        self.eps = float(state["eps"])
        self.momentum = float(state["momentum"])
        self._unpack_buffers(state, "square_avg", self.square_avg)
        self._unpack_buffers(state, "momentum_buffer", self.momentum_buffer)


class AdamW(Optimizer):

    def __init__(self, parameters, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        super().__init__(parameters)
        beta1, beta2 = betas
        if lr <= 0 or not (0 <= beta1 < 1 and 0 <= beta2 < 1) or eps <= 0 or weight_decay < 0:
            raise ValueError("invalid AdamW hyperparameters")
        self.lr = float(lr)
        self.beta1, self.beta2 = float(beta1), float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        self.t += 1
        for p, m, v in zip(self.parameters, self.m, self.v):
            if p.grad is None:
                continue
            if self.weight_decay > 0:
                p.data -= self.lr * self.weight_decay * p.data
            m *= self.beta1
            m += (1.0 - self.beta1) * p.grad
            v *= self.beta2
            v += (1.0 - self.beta2) * (p.grad ** 2)
            m_hat = m / (1.0 - self.beta1 ** self.t)
            v_hat = v / (1.0 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def state_dict(self) -> dict:
        state = {
            "lr": np.array(self.lr), "beta1": np.array(self.beta1), "beta2": np.array(self.beta2),
            "eps": np.array(self.eps), "weight_decay": np.array(self.weight_decay), "t": np.array(self.t),
        }
        self._pack_buffers(state, "m", self.m)
        self._pack_buffers(state, "v", self.v)
        return state

    def load_state_dict(self, state: dict) -> None:
        self.lr = float(state["lr"])
        self.beta1 = float(state["beta1"])
        self.beta2 = float(state["beta2"])
        self.eps = float(state["eps"])
        self.weight_decay = float(state["weight_decay"])
        self.t = int(state["t"])
        self._unpack_buffers(state, "m", self.m)
        self._unpack_buffers(state, "v", self.v)


def clip_grad_norm_(parameters: Iterable[Tensor], max_norm: float) -> float:
    parameters = [p for p in parameters if p.grad is not None]
    total_sq = 0.0
    for p in parameters:
        total_sq += float(np.sum(p.grad ** 2))
    total_norm = float(np.sqrt(total_sq))
    if max_norm > 0 and total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-12)
        for p in parameters:
            p.grad *= scale
    return total_norm


class _LRScheduler:
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.last_epoch = 0

    def step(self):
        raise NotImplementedError

    def get_last_lr(self) -> float:
        return self.optimizer.lr


class StepLR(_LRScheduler):

    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1):
        super().__init__(optimizer)
        if step_size <= 0 or not 0 < gamma <= 1:
            raise ValueError("step_size must be > 0 and 0 < gamma <= 1")
        self.step_size = step_size
        self.gamma = gamma

    def step(self):
        self.last_epoch += 1
        if self.last_epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma


class CosineAnnealingLR(_LRScheduler):

    def __init__(self, optimizer: Optimizer, t_max: int, eta_min: float = 0.0):
        super().__init__(optimizer)
        if t_max <= 0 or eta_min < 0:
            raise ValueError("t_max must be > 0 and eta_min must be >= 0")
        self.t_max = t_max
        self.eta_min = eta_min

    def step(self):
        self.last_epoch += 1
        progress = min(self.last_epoch, self.t_max) / self.t_max
        self.optimizer.lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1 + np.cos(np.pi * progress))


class LinearWarmupLR(_LRScheduler):

    def __init__(self, optimizer: Optimizer, warmup_steps: int):
        super().__init__(optimizer)
        if warmup_steps <= 0:
            raise ValueError("warmup_steps must be > 0")
        self.warmup_steps = warmup_steps

    def step(self):
        self.last_epoch += 1
        fraction = min(self.last_epoch, self.warmup_steps) / self.warmup_steps
        self.optimizer.lr = self.base_lr * fraction
