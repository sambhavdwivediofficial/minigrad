import numpy as np
from minigrad import Tensor, Linear, Sequential, Tanh, Conv2d, explain_nan, explain_gradient, GradientDebugger


def demo_explain_nan_log_domain_error():
    print("--- explain_nan: a negative value reached log() ---")
    x = Tensor([0.8, -0.3, 1.5], requires_grad=True)
    with np.errstate(invalid="ignore"):
        loss = (x.log() * 2.0).sum()
    print(explain_nan(loss).summary())
    print()


def demo_explain_nan_division_by_zero():
    print("--- explain_nan: a batch-normalization-style division by a zero variance ---")
    values = Tensor([5.0, 5.0, 5.0, 5.0], requires_grad=True)
    variance = Tensor([0.0])
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = ((values - values.mean()) / variance).sum()
    print(explain_nan(normalized).summary())
    print()


def demo_explain_nan_clean_model():
    print("--- explain_nan: a healthy model, nothing wrong ---")
    model = Sequential(Linear(3, 8, seed=1), Tanh(), Linear(8, 1, seed=2))
    x = Tensor(np.random.default_rng(0).standard_normal((4, 3)))
    loss = (model(x) ** 2).sum()
    print(explain_nan(loss).summary())
    print()


def demo_explain_gradient_simple_chain():
    print("--- explain_gradient: tanh then scale then sum ---")
    x = Tensor([0.5, -0.3, 1.2], requires_grad=True)
    derivation = explain_gradient(lambda t: (t.tanh() * 3.0).sum(), x)
    print(derivation.summary())
    print()


def demo_explain_gradient_through_conv2d():
    print("--- explain_gradient: through a real Conv2d layer ---")
    conv = Conv2d(1, 2, kernel_size=3, padding=1, seed=0)
    x = Tensor(np.random.default_rng(1).standard_normal((1, 1, 5, 5)), requires_grad=True)
    derivation = explain_gradient(lambda t: (conv(t) ** 2).sum(), x)
    print(derivation.summary())
    print()


def demo_time_travel_debugger_finds_an_exploding_gradient():
    print("--- GradientDebugger: find exactly which op exploded the gradient ---")
    x = Tensor([0.01, 0.02], requires_grad=True)
    loss = (x.exp() * 1000.0).sum()
    debugger = GradientDebugger(loss)
    jump = debugger.step_until_magnitude_jump(factor=5.0)
    print(f"Stepped {debugger.pointer + 1} of {debugger.total_steps} ops before finding a jump.")
    print(jump.summary())
    print()


def demo_time_travel_debugger_what_if_injection():
    print("--- GradientDebugger: what-if — inject a different gradient mid-replay ---")
    model = Sequential(Linear(2, 3, seed=1), Tanh(), Linear(3, 1, seed=2))
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    loss = (model(x) ** 2).sum()

    debugger = GradientDebugger(loss)
    debugger.step()
    debugger.step()
    next_node = debugger.peek_next_node()
    original = debugger.peek_grad(next_node)
    print(f"About to process a '{next_node._op}' node carrying gradient {original}.")
    debugger.inject(next_node, original * 100.0)
    print("Injected a gradient 100x larger and continued stepping...")
    debugger.step_all()
    print(f"Resulting dL/dx: {x.grad} (100x what it would have been without the injection).")


if __name__ == "__main__":
    demo_explain_nan_log_domain_error()
    demo_explain_nan_division_by_zero()
    demo_explain_nan_clean_model()
    demo_explain_gradient_simple_chain()
    demo_explain_gradient_through_conv2d()
    demo_time_travel_debugger_finds_an_exploding_gradient()
    demo_time_travel_debugger_what_if_injection()
