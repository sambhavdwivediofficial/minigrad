from __future__ import annotations

import numpy as np
from .tensor import Tensor, stack
from .nn import Module


class RNNCell(Module):
    def __init__(self, input_size: int, hidden_size: int, seed: int | None = None):
        super().__init__()
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("input_size and hidden_size must be positive")
        self.input_size = input_size
        self.hidden_size = hidden_size
        rng = np.random.default_rng(seed)
        scale_x = np.sqrt(1.0 / input_size)
        scale_h = np.sqrt(1.0 / hidden_size)
        self.w_xh = Tensor(rng.normal(0.0, scale_x, (input_size, hidden_size)), requires_grad=True)
        self.w_hh = Tensor(rng.normal(0.0, scale_h, (hidden_size, hidden_size)), requires_grad=True)
        self.b_h = Tensor(np.zeros((1, hidden_size)), requires_grad=True)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        return (x @ self.w_xh + h @ self.w_hh + self.b_h).tanh()


class LSTMCell(Module):
    def __init__(self, input_size: int, hidden_size: int, seed: int | None = None):
        super().__init__()
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("input_size and hidden_size must be positive")
        self.input_size = input_size
        self.hidden_size = hidden_size
        rng = np.random.default_rng(seed)
        scale_x = np.sqrt(1.0 / input_size)
        scale_h = np.sqrt(1.0 / hidden_size)

        def xw():
            return Tensor(rng.normal(0.0, scale_x, (input_size, hidden_size)), requires_grad=True)

        def hw():
            return Tensor(rng.normal(0.0, scale_h, (hidden_size, hidden_size)), requires_grad=True)

        self.w_xi, self.w_hi, self.b_i = xw(), hw(), Tensor(np.zeros((1, hidden_size)), requires_grad=True)
        self.w_xf, self.w_hf, self.b_f = xw(), hw(), Tensor(np.ones((1, hidden_size)), requires_grad=True)
        self.w_xg, self.w_hg, self.b_g = xw(), hw(), Tensor(np.zeros((1, hidden_size)), requires_grad=True)
        self.w_xo, self.w_ho, self.b_o = xw(), hw(), Tensor(np.zeros((1, hidden_size)), requires_grad=True)

    def forward(self, x: Tensor, state) -> tuple:
        h, c = state
        i = (x @ self.w_xi + h @ self.w_hi + self.b_i).sigmoid()
        f = (x @ self.w_xf + h @ self.w_hf + self.b_f).sigmoid()
        g = (x @ self.w_xg + h @ self.w_hg + self.b_g).tanh()
        o = (x @ self.w_xo + h @ self.w_ho + self.b_o).sigmoid()
        c_new = f * c + i * g
        h_new = o * c_new.tanh()
        return h_new, c_new


class GRUCell(Module):
    def __init__(self, input_size: int, hidden_size: int, seed: int | None = None):
        super().__init__()
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("input_size and hidden_size must be positive")
        self.input_size = input_size
        self.hidden_size = hidden_size
        rng = np.random.default_rng(seed)
        scale_x = np.sqrt(1.0 / input_size)
        scale_h = np.sqrt(1.0 / hidden_size)

        def xw():
            return Tensor(rng.normal(0.0, scale_x, (input_size, hidden_size)), requires_grad=True)

        def hw():
            return Tensor(rng.normal(0.0, scale_h, (hidden_size, hidden_size)), requires_grad=True)

        self.w_xz, self.w_hz, self.b_z = xw(), hw(), Tensor(np.zeros((1, hidden_size)), requires_grad=True)
        self.w_xr, self.w_hr, self.b_r = xw(), hw(), Tensor(np.zeros((1, hidden_size)), requires_grad=True)
        self.w_xn, self.w_hn, self.b_n = xw(), hw(), Tensor(np.zeros((1, hidden_size)), requires_grad=True)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        z = (x @ self.w_xz + h @ self.w_hz + self.b_z).sigmoid()
        r = (x @ self.w_xr + h @ self.w_hr + self.b_r).sigmoid()
        n = (x @ self.w_xn + (r * h) @ self.w_hn + self.b_n).tanh()
        return (Tensor(1.0) - z) * n + z * h


class _SequenceWrapper(Module):
    def __init__(self, cell: Module, hidden_size: int):
        super().__init__()
        self.cell = cell
        self.hidden_size = hidden_size

    def _initial_state(self, batch: int):
        raise NotImplementedError

    def _step(self, x_t: Tensor, state):
        raise NotImplementedError

    def _state_to_output(self, state):
        raise NotImplementedError

    def forward(self, x: Tensor, initial_state=None):
        batch, seq_len, _ = x.shape
        state = self._initial_state(batch) if initial_state is None else initial_state
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            state = self._step(x_t, state)
            outputs.append(self._state_to_output(state))
        return stack(outputs, axis=1), state


class RNN(_SequenceWrapper):
    def __init__(self, input_size: int, hidden_size: int, seed: int | None = None):
        super().__init__(RNNCell(input_size, hidden_size, seed=seed), hidden_size)

    def _initial_state(self, batch: int):
        return Tensor(np.zeros((batch, self.hidden_size)))

    def _step(self, x_t: Tensor, state):
        return self.cell(x_t, state)

    def _state_to_output(self, state):
        return state


class LSTM(_SequenceWrapper):
    def __init__(self, input_size: int, hidden_size: int, seed: int | None = None):
        super().__init__(LSTMCell(input_size, hidden_size, seed=seed), hidden_size)

    def _initial_state(self, batch: int):
        zeros = Tensor(np.zeros((batch, self.hidden_size)))
        return zeros, Tensor(np.zeros((batch, self.hidden_size)))

    def _step(self, x_t: Tensor, state):
        return self.cell(x_t, state)

    def _state_to_output(self, state):
        return state[0]


class GRU(_SequenceWrapper):
    def __init__(self, input_size: int, hidden_size: int, seed: int | None = None):
        super().__init__(GRUCell(input_size, hidden_size, seed=seed), hidden_size)

    def _initial_state(self, batch: int):
        return Tensor(np.zeros((batch, self.hidden_size)))

    def _step(self, x_t: Tensor, state):
        return self.cell(x_t, state)

    def _state_to_output(self, state):
        return state
