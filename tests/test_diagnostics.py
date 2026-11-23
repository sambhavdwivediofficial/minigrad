import numpy as np
import pytest
from minigrad import (
    Tensor, explain_nan, explain_gradient, Conv2d, Linear, Sequential, Tanh,
    GradientDebugger,
)


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


def _build_small_model_and_loss(seed=0):
    rng = np.random.default_rng(seed)
    model = Sequential(Linear(3, 4, seed=1), Tanh(), Linear(4, 1, seed=2))
    x = Tensor(rng.standard_normal((2, 3)), requires_grad=True)
    loss = (model(x) ** 2).sum()
    return model, x, loss


def test_debugger_step_all_matches_normal_backward_exactly():
    model_a, x_a, loss_a = _build_small_model_and_loss()
    GradientDebugger(loss_a).step_all()

    model_b, x_b, loss_b = _build_small_model_and_loss()
    x_b.data[...] = x_a.data
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        p_b.data[...] = p_a.data
    loss_b = (model_b(x_b) ** 2).sum()
    loss_b.backward()

    np.testing.assert_allclose(x_a.grad, x_b.grad)
    np.testing.assert_allclose(model_a.layers[0].weight.grad, model_b.layers[0].weight.grad)


def test_debugger_total_steps_matches_graph_size():
    _, _, loss = _build_small_model_and_loss()
    debugger = GradientDebugger(loss)
    records = debugger.step_all()
    assert len(records) == debugger.total_steps
    assert debugger.is_done()
    assert debugger.remaining() == 0


def test_debugger_first_step_is_the_output_itself():
    _, _, loss = _build_small_model_and_loss()
    debugger = GradientDebugger(loss)
    first = debugger.step()
    assert first.index == 0
    assert first.node_op == loss._op


def test_debugger_last_step_is_a_leaf():
    _, _, loss = _build_small_model_and_loss()
    debugger = GradientDebugger(loss)
    records = debugger.step_all()
    assert records[-1].parent_ops == []
    assert "leaf" in records[-1].summary() or "nothing further" in records[-1].summary()


def test_debugger_undo_then_redo_is_deterministic():
    _, _, loss = _build_small_model_and_loss()
    debugger = GradientDebugger(loss)
    debugger.step()
    debugger.step()
    third = debugger.step()
    pointer_before_undo = debugger.pointer
    debugger.undo()
    assert debugger.pointer == pointer_before_undo - 1
    third_again = debugger.step()
    assert third.node_op == third_again.node_op
    np.testing.assert_allclose(third.grad_after, third_again.grad_after)


def test_debugger_undo_raises_with_empty_history():
    _, _, loss = _build_small_model_and_loss()
    debugger = GradientDebugger(loss)
    with pytest.raises(IndexError):
        debugger.undo()


def test_debugger_step_raises_after_completion():
    _, _, loss = _build_small_model_and_loss()
    debugger = GradientDebugger(loss)
    debugger.step_all()
    with pytest.raises(StopIteration):
        debugger.step()


def test_debugger_reset_allows_a_fresh_replay():
    _, _, loss = _build_small_model_and_loss()
    debugger = GradientDebugger(loss)
    first_pass = debugger.step_all()
    debugger.reset()
    assert debugger.pointer == -1
    assert not debugger.is_done()
    second_pass = debugger.step_all()
    assert len(first_pass) == len(second_pass)
    for a, b in zip(first_pass, second_pass):
        assert a.node_op == b.node_op


def test_debugger_inject_changes_downstream_gradient_through_a_linear_path():
    model, x, loss = _build_small_model_and_loss()
    debugger = GradientDebugger(loss)
    debugger.step()
    debugger.step()
    next_node = debugger.peek_next_node()
    original = debugger.peek_grad(next_node)
    debugger.inject(next_node, original * 100.0)
    debugger.step_all()

    model2, x2, loss2 = _build_small_model_and_loss()
    x2.data[...] = x.data
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        p2.data[...] = p1.data
    loss2 = (model2(x2) ** 2).sum()
    loss2.backward()

    np.testing.assert_allclose(x.grad, x2.grad * 100.0, rtol=1e-5)


def test_debugger_inject_into_already_processed_node_raises():
    _, _, loss = _build_small_model_and_loss()
    debugger = GradientDebugger(loss)
    debugger.step()
    stale_node = debugger.current_node()
    with pytest.raises(ValueError):
        debugger.inject(stale_node, np.zeros(stale_node.data.shape))


def test_debugger_step_until_magnitude_jump_finds_exp_explosion():
    x = Tensor([0.01, 0.01], requires_grad=True)
    y = (x.exp() * 1000.0).sum()
    debugger = GradientDebugger(y)
    record = debugger.step_until_magnitude_jump(factor=1.5)
    assert record is not None
    assert record.node_op == "exp"


def test_debugger_step_until_magnitude_jump_returns_none_for_smooth_graph():
    x = Tensor([0.5, -0.5], requires_grad=True)
    y = (x * 2.0).sum()
    debugger = GradientDebugger(y)
    record = debugger.step_until_magnitude_jump(factor=1000.0)
    assert record is None
    assert debugger.is_done()


def test_debugger_rejects_non_scalar_output_without_explicit_gradient():
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x * 2.0
    with pytest.raises(ValueError):
        GradientDebugger(y)


def test_debugger_timeline_is_empty_before_any_steps():
    _, _, loss = _build_small_model_and_loss()
    debugger = GradientDebugger(loss)
    assert "no steps" in debugger.timeline()
