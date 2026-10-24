import numpy as np
import pytest
from minigrad import (
    Tensor, gradcheck, gradcheck_multi,
    Conv2d, MaxPool2d, AvgPool2d, Embedding, pad2d,
    Sequential, ReLU, Flatten, Linear,
)


def test_pad2d_shape_and_gradcheck():
    x = Tensor(np.random.randn(2, 3, 5, 5), requires_grad=True)
    out = pad2d(x, 2)
    assert out.shape == (2, 3, 9, 9)
    assert gradcheck(lambda t: (pad2d(t, 2) ** 2).sum(), x).passed


def test_pad2d_zero_padding_is_identity():
    x = Tensor(np.random.randn(2, 3, 4, 4))
    out = pad2d(x, 0)
    np.testing.assert_allclose(out.data, x.data)


@pytest.mark.parametrize(
    "in_ch, out_ch, kernel, stride, padding, h, w",
    [
        (3, 4, 3, 1, 1, 8, 8),
        (3, 4, 3, 2, 1, 9, 9),
        (1, 2, 5, 1, 0, 7, 7),
    ],
)
def test_conv2d_output_shape(in_ch, out_ch, kernel, stride, padding, h, w):
    conv = Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding, seed=0)
    x = Tensor(np.random.randn(2, in_ch, h, w))
    out = conv(x)
    expected_h = (h + 2 * padding - kernel) // stride + 1
    expected_w = (w + 2 * padding - kernel) // stride + 1
    assert out.shape == (2, out_ch, expected_h, expected_w)


def test_conv2d_gradcheck_all_inputs():
    def forward(x, w, b):
        conv = Conv2d(2, 3, kernel_size=3, stride=1, padding=1, seed=0)
        conv.weight = w
        conv.bias = b
        return (conv(x) ** 2).sum()

    x = Tensor(np.random.randn(2, 2, 5, 5), requires_grad=True)
    w = Tensor(np.random.randn(3, 2, 3, 3) * 0.1, requires_grad=True)
    b = Tensor(np.zeros((1, 3, 1, 1)), requires_grad=True)
    results = gradcheck_multi(forward, [x, w, b])
    assert all(r.passed for r in results), results


def test_conv2d_rejects_invalid_hyperparameters():
    with pytest.raises(ValueError):
        Conv2d(0, 4, kernel_size=3)
    with pytest.raises(ValueError):
        Conv2d(3, 4, kernel_size=3, stride=0)
    with pytest.raises(ValueError):
        Conv2d(3, 4, kernel_size=3, padding=-1)


def test_maxpool2d_matches_manual_max():
    x_data = np.array([[[[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0], [9.0, 10.0, 13.0, 14.0], [11.0, 12.0, 15.0, 16.0]]]])
    x = Tensor(x_data)
    out = MaxPool2d(2)(x)
    expected = np.array([[[[4.0, 8.0], [12.0, 16.0]]]])
    np.testing.assert_allclose(out.data, expected)


def test_maxpool2d_gradcheck():
    x = Tensor(np.random.randn(2, 3, 8, 8), requires_grad=True)
    assert gradcheck(lambda t: (MaxPool2d(2)(t) ** 2).sum(), x).passed


def test_avgpool2d_matches_manual_mean():
    x_data = np.array([[[[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0], [9.0, 10.0, 13.0, 14.0], [11.0, 12.0, 15.0, 16.0]]]])
    x = Tensor(x_data)
    out = AvgPool2d(2)(x)
    expected = np.array([[[[2.5, 6.5], [10.5, 14.5]]]])
    np.testing.assert_allclose(out.data, expected)


def test_avgpool2d_gradcheck():
    x = Tensor(np.random.randn(2, 3, 8, 8), requires_grad=True)
    assert gradcheck(lambda t: (AvgPool2d(2)(t) ** 2).sum(), x).passed


def test_embedding_output_shape_and_values():
    emb = Embedding(10, 4, seed=0)
    idx = np.array([[1, 2, 3], [4, 5, 1]])
    out = emb(idx)
    assert out.shape == (2, 3, 4)
    np.testing.assert_allclose(out.data[0, 0], emb.weight.data[1])
    np.testing.assert_allclose(out.data[1, 2], emb.weight.data[1])


def test_embedding_gradcheck():
    def forward(w):
        emb = Embedding(6, 3, seed=0)
        emb.weight = w
        return (emb(np.array([0, 1, 2, 3]))).sum()

    w = Tensor(np.random.randn(6, 3), requires_grad=True)
    assert gradcheck(forward, w).passed


def test_embedding_repeated_indices_accumulate_gradient():
    emb = Embedding(5, 2, seed=0)
    emb.weight = Tensor(np.zeros((5, 2)), requires_grad=True)
    out = emb(np.array([0, 0, 1]))
    out.sum().backward()
    np.testing.assert_allclose(emb.weight.grad, np.array([[2.0, 2.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]))


def test_embedding_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        Embedding(0, 4)
    with pytest.raises(ValueError):
        Embedding(4, 0)


def test_tiny_cnn_end_to_end_forward_shape():
    model = Sequential(
        Conv2d(1, 4, kernel_size=3, padding=1, seed=1),
        ReLU(),
        MaxPool2d(2),
        Conv2d(4, 8, kernel_size=3, padding=1, seed=2),
        ReLU(),
        MaxPool2d(2),
        Flatten(),
        Linear(8 * 2 * 2, 3, seed=3),
    )
    x = Tensor(np.random.randn(5, 1, 8, 8))
    out = model(x)
    assert out.shape == (5, 3)
