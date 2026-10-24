import numpy as np
import pytest
from minigrad import Tensor, gradcheck, gradcheck_multi, RNNCell, LSTMCell, GRUCell, RNN, LSTM, GRU


def test_rnn_cell_gradcheck_and_shape():
    cell = RNNCell(3, 4, seed=0)
    x = Tensor(np.random.randn(5, 3), requires_grad=True)
    h = Tensor(np.random.randn(5, 4), requires_grad=True)
    out = cell(x, h)
    assert out.shape == (5, 4)
    results = gradcheck_multi(lambda xx, hh: (cell(xx, hh) ** 2).sum(), [x, h])
    assert all(r.passed for r in results), results


def test_rnn_cell_rejects_invalid_sizes():
    with pytest.raises(ValueError):
        RNNCell(0, 4)
    with pytest.raises(ValueError):
        RNNCell(3, 0)


def test_lstm_cell_gradcheck_and_shapes():
    cell = LSTMCell(3, 4, seed=0)
    x = Tensor(np.random.randn(5, 3), requires_grad=True)
    h = Tensor(np.random.randn(5, 4), requires_grad=True)
    c = Tensor(np.random.randn(5, 4), requires_grad=True)
    h_new, c_new = cell(x, (h, c))
    assert h_new.shape == (5, 4) and c_new.shape == (5, 4)

    def forward(xx, hh, cc):
        hn, cn = cell(xx, (hh, cc))
        return (hn ** 2).sum() + (cn ** 2).sum()

    results = gradcheck_multi(forward, [x, h, c])
    assert all(r.passed for r in results), results


def test_lstm_cell_forget_gate_bias_starts_near_one():
    cell = LSTMCell(3, 4, seed=0)
    np.testing.assert_allclose(cell.b_f.data, np.ones((1, 4)))


def test_gru_cell_gradcheck_and_shape():
    cell = GRUCell(3, 4, seed=0)
    x = Tensor(np.random.randn(5, 3), requires_grad=True)
    h = Tensor(np.random.randn(5, 4), requires_grad=True)
    out = cell(x, h)
    assert out.shape == (5, 4)
    results = gradcheck_multi(lambda xx, hh: (cell(xx, hh) ** 2).sum(), [x, h])
    assert all(r.passed for r in results), results


def test_rnn_sequence_shape_and_gradcheck():
    rnn = RNN(3, 4, seed=0)
    x = Tensor(np.random.randn(2, 5, 3), requires_grad=True)
    out, h_final = rnn(x)
    assert out.shape == (2, 5, 4)
    assert h_final.shape == (2, 4)
    assert gradcheck(lambda t: (rnn(t)[0] ** 2).sum(), x).passed


def test_lstm_sequence_shape_and_gradcheck():
    lstm = LSTM(3, 4, seed=0)
    x = Tensor(np.random.randn(2, 5, 3), requires_grad=True)
    out, (h_final, c_final) = lstm(x)
    assert out.shape == (2, 5, 4)
    assert h_final.shape == (2, 4) and c_final.shape == (2, 4)
    assert gradcheck(lambda t: (lstm(t)[0] ** 2).sum(), x).passed


def test_gru_sequence_shape_and_gradcheck():
    gru = GRU(3, 4, seed=0)
    x = Tensor(np.random.randn(2, 5, 3), requires_grad=True)
    out, h_final = gru(x)
    assert out.shape == (2, 5, 4)
    assert gradcheck(lambda t: (gru(t)[0] ** 2).sum(), x).passed


def test_lstm_can_memorize_a_short_repeating_sequence():
    from minigrad import Embedding, Linear, Adam, softmax_cross_entropy

    text = "abab" * 8
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text])
    seq_len = 3
    x = np.array([data[i : i + seq_len] for i in range(len(data) - seq_len)])
    y = np.array([data[i + 1 : i + seq_len + 1] for i in range(len(data) - seq_len)])

    emb = Embedding(len(chars), 6, seed=1)
    lstm = LSTM(6, 8, seed=2)
    head = Linear(8, len(chars), seed=3)
    params = list(emb.parameters()) + list(lstm.parameters()) + list(head.parameters())
    optimizer = Adam(params, lr=0.05)

    for _ in range(120):
        optimizer.zero_grad()
        hidden, _ = lstm(emb(x))
        batch, sl, hid = hidden.shape
        logits = head(hidden.reshape(batch * sl, hid))
        loss = softmax_cross_entropy(logits, y.reshape(-1))
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.1
    predicted = logits.data.argmax(axis=-1)
    assert (predicted == y.reshape(-1)).mean() > 0.95
