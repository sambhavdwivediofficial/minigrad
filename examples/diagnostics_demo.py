import numpy as np
from minigrad import Tensor, Linear, Sequential, Tanh, Conv2d, explain_nan, explain_gradient


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


if __name__ == "__main__":
    demo_explain_nan_log_domain_error()
    demo_explain_nan_division_by_zero()
    demo_explain_nan_clean_model()
    demo_explain_gradient_simple_chain()
    demo_explain_gradient_through_conv2d()
