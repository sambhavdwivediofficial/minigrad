import numpy as np
import pytest
from minigrad import (
    Tensor, grad, hvp, hessian, newton_step,
    Linear, Sequential, Tanh, LayerNorm, BatchNorm1d, Dropout, mse_loss,
    Conv2d, Embedding,
)
from minigrad.recurrent import RNNCell


def test_second_derivative_of_cube_matches_closed_form():
    x = Tensor([2.0, -1.0, 3.0], requires_grad=True)
    y = (x ** 3).sum()
    y.backward(create_graph=True)
    first = x.grad
    assert isinstance(first, Tensor)
    np.testing.assert_allclose(first.data, 3 * x.data ** 2)
    x.grad = None
    first.sum().backward()
    np.testing.assert_allclose(x.grad, 6 * x.data)


def test_second_derivative_of_exp_is_exp():
    x = Tensor([0.1, 0.4, -0.2], requires_grad=True)
    y = x.exp().sum()
    y.backward(create_graph=True)
    first = x.grad
    x.grad = None
    first.sum().backward()
    np.testing.assert_allclose(x.grad, np.exp(x.data), atol=1e-8)


def test_second_derivative_of_tanh_matches_closed_form():
    x = Tensor([0.5, -0.3, 1.2], requires_grad=True)
    y = x.tanh().sum()
    y.backward(create_graph=True)
    first = x.grad
    x.grad = None
    first.sum().backward()
    t = np.tanh(x.data)
    expected_second = -2 * t * (1 - t ** 2)
    np.testing.assert_allclose(x.grad, expected_second, atol=1e-8)


def test_hessian_of_quadratic_form_matches_analytical_answer():
    A = np.array([[4.0, 1.0], [1.0, 3.0]])

    def f(x):
        ax = Tensor(A) @ x.reshape(2, 1)
        return (x.reshape(1, 2) @ ax).reshape(())

    x = Tensor([1.0, 1.0])
    H = hessian(f, x)
    np.testing.assert_allclose(H, A + A.T, atol=1e-8)


def test_hvp_matches_full_hessian_times_vector():
    A = np.array([[4.0, 1.0], [1.0, 3.0]])

    def f(x):
        ax = Tensor(A) @ x.reshape(2, 1)
        return (x.reshape(1, 2) @ ax).reshape(())

    x = Tensor([1.0, 1.0])
    v = np.array([1.0, -2.0])
    result = hvp(f, x, v)
    np.testing.assert_allclose(result, (A + A.T) @ v, atol=1e-8)


def test_newton_step_reaches_quadratic_minimum_in_one_step():
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([2.0, 3.0])

    def g(x):
        ax = Tensor(A) @ x.reshape(2, 1)
        quad = (x.reshape(1, 2) @ ax).reshape(())
        lin = (x * Tensor(b)).sum()
        return quad - lin

    x0 = Tensor([5.0, -2.0])
    x1 = newton_step(g, x0, damping=0.0)
    x_star = np.linalg.solve(A + A.T, b)
    np.testing.assert_allclose(x1, x_star, atol=1e-6)


def test_grad_function_first_order():
    x = Tensor([2.0, 3.0], requires_grad=True)
    y = (x ** 2).sum()
    g = grad(y, x)
    np.testing.assert_allclose(g, 2 * x.data)


def test_grad_function_chained_for_second_order():
    x = Tensor([2.0, 3.0], requires_grad=True)
    y = (x ** 3).sum()
    g1 = grad(y, x, create_graph=True)
    g2 = grad(g1.sum(), x)
    np.testing.assert_allclose(g2, 6 * x.data)


def test_grad_rejects_non_scalar_output():
    x = Tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(ValueError):
        grad(x * 2.0, x)


def test_grad_rejects_input_without_requires_grad():
    x = Tensor([1.0, 2.0], requires_grad=False)
    y = (x * 2.0).sum()
    with pytest.raises(ValueError):
        grad(y, x)


def test_second_order_through_full_mlp_and_mse_loss():
    model = Sequential(Linear(3, 5, seed=1), Tanh(), Linear(5, 1, seed=2))
    x = Tensor(np.random.default_rng(0).standard_normal((4, 3)))
    y = Tensor(np.random.default_rng(1).standard_normal((4, 1)))
    loss = mse_loss(model(x), y)
    loss.backward(create_graph=True)
    w1_grad = model.layers[0].weight.grad
    assert isinstance(w1_grad, Tensor)
    scalar = (w1_grad ** 2).sum()
    model.layers[0].weight.grad = None
    scalar.backward()
    assert np.all(np.isfinite(model.layers[0].weight.grad))


@pytest.mark.parametrize(
    "build_layer_and_input",
    [
        lambda: (LayerNorm(4), Tensor(np.random.randn(5, 4), requires_grad=True)),
        lambda: (BatchNorm1d(4), Tensor(np.random.randn(5, 4), requires_grad=True)),
        lambda: (Dropout(0.3, seed=0), Tensor(np.random.randn(5, 4), requires_grad=True)),
    ],
)
def test_composed_layers_support_create_graph(build_layer_and_input):
    layer, x = build_layer_and_input()
    out = (layer(x) ** 2).sum()
    out.backward(create_graph=True)
    assert isinstance(x.grad, Tensor)


def test_rnn_cell_single_step_supports_create_graph():
    cell = RNNCell(3, 4, seed=0)
    x = Tensor(np.random.randn(2, 3), requires_grad=True)
    h = Tensor(np.random.randn(2, 4), requires_grad=True)
    out = (cell(x, h) ** 2).sum()
    out.backward(create_graph=True)
    assert isinstance(x.grad, Tensor)
    assert isinstance(h.grad, Tensor)


def test_getitem_raises_clear_error_under_create_graph():
    x = Tensor(np.random.randn(5), requires_grad=True)
    out = (x[np.array([0, 1, 2])] ** 2).sum()
    with pytest.raises(NotImplementedError):
        out.backward(create_graph=True)


def test_conv2d_raises_clear_error_under_create_graph():
    conv = Conv2d(1, 2, kernel_size=3, padding=1, seed=0)
    x = Tensor(np.random.randn(1, 1, 5, 5), requires_grad=True)
    out = (conv(x) ** 2).sum()
    with pytest.raises(NotImplementedError):
        out.backward(create_graph=True)


def test_embedding_raises_clear_error_under_create_graph():
    emb = Embedding(10, 4, seed=0)
    out = (emb(np.array([1, 2, 3])) ** 2).sum()
    with pytest.raises(NotImplementedError):
        out.backward(create_graph=True)


def test_hessian_and_hvp_reject_non_1d_input():
    def f(x):
        return (x ** 2).sum()

    x2d = Tensor(np.random.randn(2, 2))
    with pytest.raises(ValueError):
        hessian(f, x2d)
    with pytest.raises(ValueError):
        hvp(f, x2d, np.zeros((2, 2)))
