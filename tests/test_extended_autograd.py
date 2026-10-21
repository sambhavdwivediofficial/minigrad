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
