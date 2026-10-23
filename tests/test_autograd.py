import numpy as np
import pytest
from minigrad import Tensor


def test_add_and_multiply_gradients():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    loss = ((x * y) + x).sum()
    loss.backward()
    np.testing.assert_allclose(x.grad, np.array([5.0, 6.0, 7.0]))
    np.testing.assert_allclose(y.grad, np.array([1.0, 2.0, 3.0]))


def test_broadcast_gradient():
    x = Tensor(np.ones((3, 2)), requires_grad=True)
    bias = Tensor(np.array([[2.0, 3.0]]), requires_grad=True)
    (x + bias).sum().backward()
    np.testing.assert_allclose(x.grad, np.ones((3, 2)))
    np.testing.assert_allclose(bias.grad, np.array([[3.0, 3.0]]))


def test_matmul_gradient():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    w = Tensor([[2.0], [1.0]], requires_grad=True)
    (x @ w).sum().backward()
    np.testing.assert_allclose(x.grad, np.array([[2.0, 1.0], [2.0, 1.0]]))
    np.testing.assert_allclose(w.grad, np.array([[4.0], [6.0]]))


def test_tanh_gradient():
    x = Tensor([0.1, -0.4], requires_grad=True)
    x.tanh().sum().backward()
    expected = 1 - np.tanh(x.data) ** 2
    np.testing.assert_allclose(x.grad, expected, rtol=1e-7, atol=1e-7)


def test_finite_difference_gradcheck():
    from minigrad import gradcheck
    x = Tensor([0.2, -0.7, 1.1], requires_grad=True)
    result = gradcheck(lambda value: ((value * value) + value.tanh()).sum(), x)
    assert result.passed, result



def test_shared_tensor_contributes_through_multiple_paths():
    x = Tensor([2.0], requires_grad=True)
    y = x * x + x
    loss = y.sum()
    loss.backward()
    np.testing.assert_allclose(x.grad, np.array([2.0 * 2.0 + 1.0]))


def test_shared_tensor_three_paths():
    x = Tensor([1.5, -0.5], requires_grad=True)
    loss = (x * 2.0).sum() + (x * x).sum() + x.tanh().sum()
    loss.backward()
    expected = 2.0 + 2.0 * x.data + (1 - np.tanh(x.data) ** 2)
    np.testing.assert_allclose(x.grad, expected, rtol=1e-7, atol=1e-7)


def test_repeated_backward_accumulates_until_zero_grad():
    x = Tensor([3.0], requires_grad=True)
    (x * x).sum().backward()
    first = x.grad.copy()
    (x * x).sum().backward()
    second = x.grad.copy()
    np.testing.assert_allclose(second, first * 2, err_msg="grad should accumulate across backward() calls")
    x.zero_grad()
    (x * x).sum().backward()
    np.testing.assert_allclose(x.grad, first, err_msg="zero_grad() should reset accumulation")


def test_backward_without_requires_grad_raises():
    x = Tensor([1.0, 2.0], requires_grad=False)
    with pytest.raises(RuntimeError):
        x.sum().backward()


def test_backward_shape_mismatch_raises():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = x * 2.0
    with pytest.raises(ValueError):
        y.backward(np.ones((3, 3)))


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((3, 4), (1, 4)),
        ((3, 4), (3, 1)),
        ((3, 4), ()),
        ((2, 3, 4), (1, 3, 4)),
        ((2, 3, 4), (2, 1, 4)),
        ((2, 3, 4), (1, 1, 4)),
        ((5,), (1,)),
    ],
)
def test_broadcasting_shape_sweep_add(shape_a, shape_b):
    rng = np.random.default_rng(0)
    a = Tensor(rng.normal(size=shape_a), requires_grad=True)
    b = Tensor(rng.normal(size=shape_b) if shape_b else rng.normal(), requires_grad=True)
    out = (a + b)
    assert out.shape == np.broadcast_shapes(shape_a, shape_b)
    out.sum().backward()
    assert a.grad.shape == shape_a
    assert b.grad.shape == (shape_b if shape_b else ())
    np.testing.assert_allclose(a.grad, np.ones(shape_a))


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((3, 4), (1, 4)),
        ((3, 4), (3, 1)),
        ((2, 3, 4), (1, 3, 4)),
    ],
)
def test_broadcasting_shape_sweep_mul(shape_a, shape_b):
    rng = np.random.default_rng(1)
    a = Tensor(rng.normal(size=shape_a), requires_grad=True)
    b = Tensor(rng.normal(size=shape_b), requires_grad=True)
    from minigrad import gradcheck_multi

    results = gradcheck_multi(lambda aa, bb: (aa * bb).sum(), [a, b])
    assert all(r.passed for r in results), results
