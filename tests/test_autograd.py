import numpy as np
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
