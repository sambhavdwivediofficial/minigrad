import numpy as np
import pytest
from minigrad import (
    Tensor, gradcheck, gradcheck_multi,
    Conv2d, MaxPool2d, AvgPool2d, Embedding, pad2d,
    Sequential, ReLU, Flatten, Linear,
)


@pytest.mark.parametrize("kernel_size, stride", [(2, 1), (3, 1), (2, 2), (3, 2), (3, 3)])
def test_unfold2d_matches_manual_sliding_window(kernel_size, stride):
    x_data = np.random.default_rng(0).standard_normal((2, 3, 9, 9))
    x = Tensor(x_data)
    windows = x.unfold2d(kernel_size, stride=stride)
    h_out = (9 - kernel_size) // stride + 1
    w_out = (9 - kernel_size) // stride + 1
    assert windows.shape == (2, 3, h_out, w_out, kernel_size, kernel_size)
    manual = np.zeros(windows.shape)
    for i in range(h_out):
        for j in range(w_out):
            manual[:, :, i, j] = x_data[:, :, i * stride : i * stride + kernel_size, j * stride : j * stride + kernel_size]
    np.testing.assert_allclose(windows.data, manual)


@pytest.mark.parametrize("kernel_size, stride", [(2, 1), (3, 1), (2, 2), (3, 2)])
def test_unfold2d_gradcheck(kernel_size, stride):
    x = Tensor(np.random.default_rng(1).standard_normal((2, 2, 7, 7)), requires_grad=True)
    assert gradcheck(lambda t: (t.unfold2d(kernel_size, stride=stride) ** 2).sum(), x).passed


def test_unfold2d_is_safe_on_non_contiguous_input():
    x = Tensor(np.asfortranarray(np.random.default_rng(2).standard_normal((2, 3, 7, 7))), requires_grad=True)
    windows = x.unfold2d(3, stride=1)
    manual = np.zeros((2, 3, 5, 5, 3, 3))
    for i in range(5):
        for j in range(5):
            manual[:, :, i, j] = x.data[:, :, i : i + 3, j : j + 3]
    np.testing.assert_allclose(windows.data, manual)
    assert gradcheck(lambda t: (t.unfold2d(3, stride=1) ** 2).sum(), x).passed


def test_unfold2d_rejects_non_4d_input():
    with pytest.raises(ValueError):
        Tensor(np.random.randn(5)).unfold2d(2)


def test_unfold2d_rejects_oversized_kernel():
    with pytest.raises(ValueError):
        Tensor(np.random.randn(1, 1, 5, 5)).unfold2d(10)


@pytest.mark.parametrize("kernel_size, stride", [(2, 1), (3, 1), (2, 2), (3, 2)])
def test_unfold2d_second_derivative_matches_numpy_mode_backward(kernel_size, stride):
    def numpy_mode_grad(xv):
        xt = Tensor(xv, requires_grad=True)
        (xt.unfold2d(kernel_size, stride=stride) ** 3).sum().backward()
        return xt.grad.copy()

    x = Tensor(np.random.default_rng(5).standard_normal((1, 1, 7, 7)), requires_grad=True)
    first = (x.unfold2d(kernel_size, stride=stride) ** 3).sum()
    first.backward(create_graph=True)
    assert isinstance(x.grad, Tensor)
    np.testing.assert_allclose(x.grad.data, numpy_mode_grad(x.data), atol=1e-8)


def test_unfold2d_second_derivative_matches_independent_finite_difference_with_overlap():
    kernel_size, stride = 3, 1

    def grad_at(xv):
        xt = Tensor(xv, requires_grad=True)
        (xt.unfold2d(kernel_size, stride=stride) ** 3).sum().backward()
        return xt.grad.copy()

    x0 = np.random.default_rng(6).standard_normal((1, 1, 5, 5))
    x = Tensor(x0, requires_grad=True)
    first = (x.unfold2d(kernel_size, stride=stride) ** 3).sum()
    first.backward(create_graph=True)
    first_grad = x.grad
    x.grad = None
    first_grad.sum().backward()
    analytic_second = x.grad.copy()

    eps = 1e-4
    for _ in range(5):
        idx = tuple(np.random.default_rng(7).integers(0, s) for s in x0.shape)
        xp, xm = x0.copy(), x0.copy()
        xp[idx] += eps
        xm[idx] -= eps
        numeric = (grad_at(xp)[idx] - grad_at(xm)[idx]) / (2 * eps)
        assert abs(numeric - analytic_second[idx]) < 1e-2


def test_conv2d_second_derivative_is_finite_through_weight_and_input():
    conv = Conv2d(2, 3, kernel_size=3, stride=1, padding=1, seed=0)
    x = Tensor(np.random.default_rng(8).standard_normal((2, 2, 6, 6)), requires_grad=True)
    loss = (conv(x) ** 2).sum()
    loss.backward(create_graph=True)
    assert isinstance(conv.weight.grad, Tensor)
    assert isinstance(x.grad, Tensor)

    weight_scalar = (conv.weight.grad ** 2).sum()
    conv.weight.grad = None
    weight_scalar.backward()
    assert np.all(np.isfinite(conv.weight.grad))


@pytest.mark.parametrize("pool_cls", [MaxPool2d, AvgPool2d])
def test_pooling_second_derivative_is_finite_with_overlapping_windows(pool_cls):
    pool = pool_cls(2, stride=1)
    x = Tensor(np.random.default_rng(9).standard_normal((1, 2, 5, 5)), requires_grad=True)
    out = (pool(x) ** 2).sum()
    out.backward(create_graph=True)
    assert isinstance(x.grad, Tensor)
    scalar = (x.grad ** 2).sum()
    x.grad = None
    scalar.backward()
    assert np.all(np.isfinite(x.grad))


def test_conv2d_matches_reference_loop_implementation():
    from minigrad import stack as _stack

    def reference_conv(conv, x):
        x_padded = x
        N, C, H, W = x_padded.shape
        k, s = conv.kernel_size, conv.stride
        h_out = (H - k) // s + 1
        w_out = (W - k) // s + 1
        patches = []
        for i in range(h_out):
            for j in range(w_out):
                patch = x_padded[:, :, i * s : i * s + k, j * s : j * s + k]
                patches.append(patch.reshape(N, C * k * k))
        col = _stack(patches, axis=1)
        weight_flat = conv.weight.reshape(conv.out_channels, C * k * k).transpose(1, 0)
        out_flat = col @ weight_flat
        return out_flat.transpose(0, 2, 1).reshape(N, conv.out_channels, h_out, w_out) + conv.bias

    conv = Conv2d(3, 4, kernel_size=3, stride=1, padding=0, seed=0)
    x = Tensor(np.random.default_rng(3).standard_normal((2, 3, 10, 10)))
    fast_out = conv(x)
    reference_out = reference_conv(conv, x)
    np.testing.assert_allclose(fast_out.data, reference_out.data, atol=1e-8)


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

    rng = np.random.default_rng(0)
    x = Tensor(rng.standard_normal((2, 2, 5, 5)), requires_grad=True)
    w = Tensor(rng.standard_normal((3, 2, 3, 3)) * 0.1, requires_grad=True)
    b = Tensor(np.zeros((1, 3, 1, 1)), requires_grad=True)
    results = gradcheck_multi(forward, [x, w, b], rtol=3e-3)
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
