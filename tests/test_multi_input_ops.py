import numpy as np
import pytest
from minigrad import Tensor, where, concatenate, stack, gradcheck, gradcheck_multi


def test_matmul_gradcheck_multi():
    rng = np.random.default_rng(0)
    a = Tensor(rng.normal(size=(3, 4)), requires_grad=True)
    b = Tensor(rng.normal(size=(4, 2)), requires_grad=True)
    results = gradcheck_multi(lambda aa, bb: (aa @ bb).sum(), [a, b])
    assert all(r.passed for r in results), results


@pytest.mark.parametrize(
    "shape_a, shape_b, expected_out_shape",
    [
        ((4, 3, 5), (4, 5, 2), (4, 3, 2)),
        ((2, 3, 4, 5), (2, 3, 5, 6), (2, 3, 4, 6)),
        ((5, 3, 4), (4, 2), (5, 3, 2)),
        ((1, 3, 4), (6, 4, 2), (6, 3, 2)),
    ],
)
def test_batched_matmul_gradcheck(shape_a, shape_b, expected_out_shape):
    rng = np.random.default_rng(1)
    a = Tensor(rng.normal(size=shape_a), requires_grad=True)
    b = Tensor(rng.normal(size=shape_b), requires_grad=True)
    out = a @ b
    assert out.shape == expected_out_shape
    results = gradcheck_multi(lambda aa, bb: (aa @ bb).sum(), [a, b])
    assert all(r.passed for r in results), results


def test_matmul_rejects_1d_operand():
    with pytest.raises(ValueError):
        Tensor(np.random.randn(3)) @ Tensor(np.random.randn(3))


def test_matmul_rejects_mismatched_inner_dimension():
    with pytest.raises(ValueError):
        Tensor(np.random.randn(3, 4)) @ Tensor(np.random.randn(5, 2))


def test_swapaxes_gradcheck_and_shape():
    x = Tensor(np.random.randn(2, 3, 4, 5), requires_grad=True)
    out = x.swapaxes(1, 2)
    assert out.shape == (2, 4, 3, 5)
    assert gradcheck(lambda t: (t.swapaxes(-1, -2) ** 2).sum(), x).passed


@pytest.mark.parametrize("axis", [0, 1])
def test_concatenate_gradcheck(axis):
    rng = np.random.default_rng(1)
    shape = (3, 4) if axis == 1 else (2, 5)
    a = Tensor(rng.normal(size=shape), requires_grad=True)
    b = Tensor(rng.normal(size=shape), requires_grad=True)
    c = Tensor(rng.normal(size=shape), requires_grad=True)
    results = gradcheck_multi(lambda aa, bb, cc: (concatenate([aa, bb, cc], axis=axis) ** 2).sum(), [a, b, c])
    assert all(r.passed for r in results), results


def test_concatenate_matches_numpy_forward():
    a = Tensor([[1.0, 2.0]], requires_grad=True)
    b = Tensor([[3.0, 4.0]], requires_grad=True)
    out = concatenate([a, b], axis=0)
    np.testing.assert_allclose(out.data, np.array([[1.0, 2.0], [3.0, 4.0]]))


@pytest.mark.parametrize("axis", [0, 1])
def test_stack_gradcheck(axis):
    rng = np.random.default_rng(2)
    a = Tensor(rng.normal(size=(4,)), requires_grad=True)
    b = Tensor(rng.normal(size=(4,)), requires_grad=True)
    results = gradcheck_multi(lambda aa, bb: (stack([aa, bb], axis=axis) ** 2).sum(), [a, b])
    assert all(r.passed for r in results), results


def test_stack_matches_numpy_forward():
    a = Tensor([1.0, 2.0], requires_grad=True)
    b = Tensor([3.0, 4.0], requires_grad=True)
    out = stack([a, b], axis=0)
    np.testing.assert_allclose(out.data, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_where_gradcheck():
    rng = np.random.default_rng(3)
    a = Tensor(rng.normal(size=(5,)), requires_grad=True)
    b = Tensor(rng.normal(size=(5,)), requires_grad=True)
    condition = np.array([True, False, True, False, True])
    results = gradcheck_multi(lambda aa, bb: (where(condition, aa, bb) ** 2).sum(), [a, b])
    assert all(r.passed for r in results), results


def test_where_selects_elementwise():
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([10.0, 20.0, 30.0], requires_grad=True)
    condition = np.array([True, False, True])
    out = where(condition, a, b)
    np.testing.assert_allclose(out.data, np.array([1.0, 20.0, 3.0]))
    out.sum().backward()
    np.testing.assert_allclose(a.grad, np.array([1.0, 0.0, 1.0]))
    np.testing.assert_allclose(b.grad, np.array([0.0, 1.0, 0.0]))
