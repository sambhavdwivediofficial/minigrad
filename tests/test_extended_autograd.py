import numpy as np
import pytest
from minigrad import Tensor, gradcheck


@pytest.mark.parametrize(
    "function, values",
    [
        (lambda value: (value / 2.7).sum(), [0.2, 1.1, 2.4]),
        (lambda value: (value / (value + 3.0)).sum(), [0.2, 1.1, 2.4]),
        (lambda value: (value ** 3.0).sum(), [-0.7, 0.4, 1.1]),
        (lambda value: (-value).sum(), [-0.7, 0.4, 1.1]),
        (lambda value: value.relu().sum(), [-0.7, 0.4, 1.1]),
        (lambda value: value.sigmoid().sum(), [-1.2, 0.3, 1.4]),
        (lambda value: value.softplus().sum(), [-1.2, 0.3, 1.4]),
        (lambda value: value.mean(), [-0.7, 0.4, 1.1]),
    ],
)
def test_extended_operation_gradients(function, values):
    value = Tensor(np.asarray(values, dtype=np.float64), requires_grad=True)
    result = gradcheck(function, value)
    assert result.passed, result


@pytest.mark.parametrize(
    "function, shape",
    [
        (lambda t: t.reshape(6).sum() * 1.3 + (t.reshape(2, 3) ** 2).sum(), (6,)),
        (lambda t: (t.reshape(3, 2).T ** 2).sum(), (6,)),
        (lambda t: (t.transpose(1, 0) ** 2).sum(), (2, 3)),
        (lambda t: (t.sum(axis=0) ** 2).sum(), (3, 4)),
        (lambda t: (t.sum(axis=1, keepdims=True) ** 2).sum(), (3, 4)),
        (lambda t: (t.mean(axis=0) ** 2).sum(), (3, 4)),
        (lambda t: t.exp().sum(), (4,)),
        (lambda t: t.abs().sum(), (4,)),
        (lambda t: (t.max(axis=1) ** 2).sum(), (3, 5)),
        (lambda t: t.max() * 3.0, (3, 5)),
        (lambda t: (t[1:3, :2] ** 2).sum(), (5, 4)),
        (lambda t: (t.log_softmax(axis=1) ** 2).sum(), (4, 5)),
        (lambda t: (t.softmax(axis=1) ** 2).sum(), (4, 5)),
    ],
)
def test_shape_and_indexing_op_gradients(function, shape):
    rng = np.random.default_rng(42)
    values = rng.normal(loc=0.0, scale=1.0, size=shape) + np.sign(rng.normal(size=shape)) * 0.5
    value = Tensor(values, requires_grad=True)
    result = gradcheck(function, value)
    assert result.passed, result


def test_log_domain_is_positive_reals_only():
    x = Tensor([1.0, -1.0, 0.0])
    with np.errstate(divide="ignore", invalid="ignore"):
        result = x.log().data
    assert np.isnan(result[1])
    assert np.isneginf(result[2])


def test_getitem_accumulates_repeated_indices():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    picked = x[np.array([0, 0, 1])]
    picked.sum().backward()
    np.testing.assert_allclose(x.grad, np.array([2.0, 1.0, 0.0]))
