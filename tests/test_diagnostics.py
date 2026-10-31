import numpy as np
import pytest
from minigrad import Tensor, explain_nan, explain_gradient, Conv2d, Linear, Sequential, Tanh


def test_explain_nan_finds_nothing_in_a_clean_graph():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    loss = (x.tanh() * 2.0).sum()
    report = explain_nan(loss)
    assert report.found is False
    assert "No NaN" in report.summary()


def test_explain_nan_pinpoints_log_of_nonpositive_value():
    x = Tensor([1.0, -2.0, 3.0], requires_grad=True)
    with np.errstate(invalid="ignore"):
        loss = (x.log() * 2.0).sum()
    report = explain_nan(loss)
    assert report.found is True
    assert report.op == "log"
    assert report.bad_count == 1
    assert report.total_count == 3
    assert "log(x)" in report.hint
    assert report.path_from_root[-1] == "log"


def test_explain_nan_pinpoints_division_by_zero():
    a = Tensor([1.0, 2.0], requires_grad=True)
    b = Tensor([2.0, 0.0], requires_grad=True)
    with np.errstate(divide="ignore"):
        loss = (a / b).sum()
    report = explain_nan(loss)
    assert report.found is True
    assert report.op == "/"
    assert "denominator" in report.hint


def test_explain_nan_pinpoints_exp_overflow():
    x = Tensor([1.0, 1000.0, 2.0], requires_grad=True)
    with np.errstate(over="ignore"):
        loss = x.exp().sum()
    report = explain_nan(loss)
    assert report.found is True
    assert report.op == "exp"
    assert "overflowed" in report.hint


def test_explain_nan_identifies_bad_leaf_input():
    x = Tensor(np.array([1.0, float("nan"), 2.0]), requires_grad=True)
    loss = (x * 2.0).sum()
    report = explain_nan(loss)
    assert report.found is True
    assert report.op == "input"
    assert "leaf tensor" in report.summary()


def test_explain_nan_in_diamond_graph_finds_the_actual_culprit():
    x = Tensor([1.0, -1.0, 2.0], requires_grad=True)
    with np.errstate(invalid="ignore"):
        a = x.log()
    b = x * 2.0
    loss = (a + b).sum()
    report = explain_nan(loss)
    assert report.op == "log"


def test_explain_nan_through_a_real_conv2d_forward_pass():
    model = Sequential(Conv2d(1, 2, kernel_size=3, padding=1, seed=0))
    x = Tensor(np.array([[[[1.0, float("nan"), 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]]), requires_grad=True)
    report = explain_nan(model(x).sum())
    assert report.found is True
    assert report.op == "input"


def test_explain_gradient_lists_correct_steps_for_simple_chain():
    x = Tensor([0.5, -0.3, 1.2], requires_grad=True)
    derivation = explain_gradient(lambda t: (t.tanh() * 2.0).sum(), x)
    ops = [op for op, _ in derivation.steps]
    assert ops == ["tanh", "*", "sum"]
    assert derivation.verified is True


def test_explain_gradient_is_itself_gradcheck_verified():
    x = Tensor(np.random.default_rng(0).standard_normal(5), requires_grad=True)
    derivation = explain_gradient(lambda t: (t ** 3).sum(), x)
    assert derivation.verified is True
    assert derivation.max_relative_error < 1e-3


def test_explain_gradient_through_conv2d_lists_unfold2d():
    conv = Conv2d(2, 3, kernel_size=3, padding=1, seed=0)
    x = Tensor(np.random.default_rng(1).standard_normal((2, 2, 5, 5)), requires_grad=True)
    derivation = explain_gradient(lambda t: (conv(t) ** 2).sum(), x)
    ops = [op for op, _ in derivation.steps]
    assert "unfold2d" in ops
    assert derivation.verified is True


def test_explain_gradient_rejects_non_scalar_output():
    x = Tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(ValueError):
        explain_gradient(lambda t: t * 2.0, x)


def test_explain_gradient_rejects_input_without_requires_grad():
    x = Tensor([1.0, 2.0], requires_grad=False)
    with pytest.raises(ValueError):
        explain_gradient(lambda t: (t * 2.0).sum(), x)


def test_explain_gradient_summary_is_readable_text():
    x = Tensor([1.0, 2.0], requires_grad=True)
    derivation = explain_gradient(lambda t: (t ** 2).sum(), x)
    text = derivation.summary()
    assert "Step 1" in text
    assert "Verification" in text
    assert "PASSED" in text
