from __future__ import annotations

import numpy as np
from .tensor import Tensor, where
from .nn import Module, Linear, LayerNorm, Embedding, ReLU


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> Tensor:
    position = np.arange(seq_len)[:, None]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    if d_model % 2 == 0:
        pe[:, 1::2] = np.cos(position * div_term)
    else:
        pe[:, 1::2] = np.cos(position * div_term)[:, : pe[:, 1::2].shape[1]]
    return Tensor(pe)


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask=None) -> Tensor:
    d_k = q.shape[-1]
    scores = (q @ k.swapaxes(-1, -2)) / (d_k ** 0.5)
    if mask is not None:
        scores = where(mask, scores, Tensor(-1e9))
    weights = scores.softmax(axis=-1)
    return weights @ v


class MultiHeadSelfAttention(Module):
    def __init__(self, d_model: int, num_heads: int, seed: int | None = None):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        rng = np.random.default_rng(seed)
        scale = np.sqrt(1.0 / d_model)
        self.w_q = Tensor(rng.normal(0.0, scale, (d_model, d_model)), requires_grad=True)
        self.w_k = Tensor(rng.normal(0.0, scale, (d_model, d_model)), requires_grad=True)
        self.w_v = Tensor(rng.normal(0.0, scale, (d_model, d_model)), requires_grad=True)
        self.w_o = Tensor(rng.normal(0.0, scale, (d_model, d_model)), requires_grad=True)

    def _split_heads(self, x: Tensor, batch: int, seq_len: int) -> Tensor:
        return x.reshape(batch, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

    def _merge_heads(self, x: Tensor, batch: int, seq_len: int) -> Tensor:
        return x.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.d_model)

    def forward(self, x: Tensor, causal: bool = False) -> Tensor:
        batch, seq_len, d_model = x.shape
        q = self._split_heads(x @ self.w_q, batch, seq_len)
        k = self._split_heads(x @ self.w_k, batch, seq_len)
        v = self._split_heads(x @ self.w_v, batch, seq_len)

        mask = np.tril(np.ones((seq_len, seq_len), dtype=bool)) if causal else None
        attn_out = scaled_dot_product_attention(q, k, v, mask)

        merged = self._merge_heads(attn_out, batch, seq_len)
        return merged @ self.w_o


class FeedForward(Module):
    def __init__(self, d_model: int, d_ff: int, seed: int | None = None):
        super().__init__()
        seed2 = None if seed is None else seed + 1
        self.linear1 = Linear(d_model, d_ff, seed=seed)
        self.activation = ReLU()
        self.linear2 = Linear(d_ff, d_model, seed=seed2)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.activation(self.linear1(x)))


class TransformerEncoderBlock(Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, seed: int | None = None):
        super().__init__()
        seed_ff = None if seed is None else seed + 10
        self.attention = MultiHeadSelfAttention(d_model, num_heads, seed=seed)
        self.norm1 = LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, seed=seed_ff)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x: Tensor, causal: bool = False) -> Tensor:
        x = self.norm1(x + self.attention(x, causal=causal))
        x = self.norm2(x + self.feed_forward(x))
        return x


class TransformerLM(Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        max_seq_len: int,
        seed: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.token_embedding = Embedding(vocab_size, d_model, seed=seed)
        self.register_buffer("positional_encoding", sinusoidal_positional_encoding(max_seq_len, d_model).data)
        self.blocks = [
            TransformerEncoderBlock(d_model, num_heads, d_ff, seed=(None if seed is None else seed + 100 * i))
            for i in range(num_layers)
        ]
        self.output_projection = Linear(d_model, vocab_size, seed=seed)

    def forward(self, token_ids) -> Tensor:
        token_ids = np.asarray(token_ids.data if isinstance(token_ids, Tensor) else token_ids, dtype=np.int64)
        batch, seq_len = token_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}")
        x = self.token_embedding(token_ids)
        pos = Tensor(self.buffer("positional_encoding")[:seq_len])
        x = x + pos
        for block in self.blocks:
            x = block(x, causal=True)
        return self.output_projection(x)
