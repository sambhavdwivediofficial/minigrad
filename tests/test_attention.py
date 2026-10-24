import numpy as np
import pytest
from minigrad import (
    Tensor, gradcheck, gradcheck_multi,
    scaled_dot_product_attention, MultiHeadSelfAttention, FeedForward,
    TransformerEncoderBlock, TransformerLM, sinusoidal_positional_encoding,
)
from minigrad.nn import LayerNorm


def test_scaled_dot_product_attention_gradcheck():
    q = Tensor(np.random.randn(5, 4), requires_grad=True)
    k = Tensor(np.random.randn(5, 4), requires_grad=True)
    v = Tensor(np.random.randn(5, 4), requires_grad=True)
    results = gradcheck_multi(lambda qq, kk, vv: (scaled_dot_product_attention(qq, kk, vv) ** 2).sum(), [q, k, v])
    assert all(r.passed for r in results), results


def test_scaled_dot_product_attention_weights_sum_to_one():
    q = Tensor(np.random.randn(4, 3))
    k = Tensor(np.random.randn(4, 3))
    d_k = q.shape[-1]
    scores = (q @ k.transpose(1, 0)) / (d_k ** 0.5)
    weights = scores.softmax(axis=-1)
    np.testing.assert_allclose(weights.data.sum(axis=-1), np.ones(4), atol=1e-8)


def test_causal_mask_gradcheck():
    q = Tensor(np.random.randn(5, 4), requires_grad=True)
    k = Tensor(np.random.randn(5, 4), requires_grad=True)
    v = Tensor(np.random.randn(5, 4), requires_grad=True)
    mask = np.tril(np.ones((5, 5), dtype=bool))
    results = gradcheck_multi(lambda qq, kk, vv: (scaled_dot_product_attention(qq, kk, vv, mask) ** 2).sum(), [q, k, v])
    assert all(r.passed for r in results), results


def test_causal_mask_blocks_future_positions():
    mha = MultiHeadSelfAttention(8, num_heads=2, seed=0)
    mha.eval()
    x1 = Tensor(np.random.randn(1, 5, 8))
    x2 = x1.detach()
    x2.data[0, 4, :] += 1000.0
    out1 = mha(x1, causal=True).data
    out2 = mha(x2, causal=True).data
    np.testing.assert_allclose(out1[0, 0], out2[0, 0], atol=1e-6)
    np.testing.assert_allclose(out1[0, 2], out2[0, 2], atol=1e-6)
    assert not np.allclose(out1[0, 4], out2[0, 4])


def test_multihead_self_attention_shape_and_gradcheck():
    mha = MultiHeadSelfAttention(8, num_heads=2, seed=0)
    x = Tensor(np.random.randn(2, 4, 8), requires_grad=True)
    out = mha(x)
    assert out.shape == (2, 4, 8)
    assert gradcheck(lambda t: (mha(t) ** 2).sum(), x).passed


def test_multihead_self_attention_rejects_bad_head_count():
    with pytest.raises(ValueError):
        MultiHeadSelfAttention(10, num_heads=3)


def test_layernorm_on_3d_sequence_input_gradcheck():
    ln = LayerNorm(8)
    x = Tensor(np.random.randn(2, 3, 8), requires_grad=True)
    assert gradcheck(lambda t: (ln(t) ** 2).sum(), x).passed


def test_feedforward_shape_and_gradcheck():
    ff = FeedForward(8, 16, seed=0)
    x = Tensor(np.random.randn(2, 4, 8), requires_grad=True)
    out = ff(x)
    assert out.shape == (2, 4, 8)
    assert gradcheck(lambda t: (ff(t) ** 2).sum(), x).passed


def test_transformer_encoder_block_shape_and_gradcheck():
    block = TransformerEncoderBlock(8, num_heads=2, d_ff=16, seed=0)
    x = Tensor(np.random.randn(2, 4, 8), requires_grad=True)
    out = block(x, causal=True)
    assert out.shape == (2, 4, 8)
    assert gradcheck(lambda t: (block(t, causal=True) ** 2).sum(), x).passed


def test_sinusoidal_positional_encoding_shape_and_bounds():
    pe = sinusoidal_positional_encoding(10, 8)
    assert pe.shape == (10, 8)
    assert np.all(pe.data >= -1.0) and np.all(pe.data <= 1.0)


def test_transformer_lm_output_shape():
    lm = TransformerLM(vocab_size=15, d_model=16, num_heads=2, d_ff=32, num_layers=2, max_seq_len=10, seed=0)
    tokens = np.random.randint(0, 15, size=(3, 6))
    logits = lm(tokens)
    assert logits.shape == (3, 6, 15)


def test_transformer_lm_can_memorize_a_short_repeating_sequence():
    from minigrad import Adam, softmax_cross_entropy

    text = "abab" * 8
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text])
    seq_len = 4
    x = np.array([data[i : i + seq_len] for i in range(len(data) - seq_len)])
    y = np.array([data[i + 1 : i + seq_len + 1] for i in range(len(data) - seq_len)])

    lm = TransformerLM(
        vocab_size=len(chars), d_model=16, num_heads=2, d_ff=32, num_layers=2, max_seq_len=seq_len, seed=1
    )
    optimizer = Adam(lm.parameters(), lr=0.03)
    for _ in range(120):
        optimizer.zero_grad()
        logits = lm(x)
        loss = softmax_cross_entropy(logits.reshape(-1, len(chars)), y.reshape(-1))
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.1
    predicted = logits.data.argmax(axis=-1)
    assert (predicted.reshape(-1) == y.reshape(-1)).mean() > 0.95
