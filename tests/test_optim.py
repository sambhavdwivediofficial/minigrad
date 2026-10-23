import numpy as np
import pytest
from minigrad import (
    Tensor, Linear, Sequential, Tanh, mse_loss,
    SGD, Adam, RMSProp, AdamW, clip_grad_norm_,
    StepLR, CosineAnnealingLR, LinearWarmupLR,
)


def _fit_xor(optimizer_cls, **kwargs):
    rng = np.random.default_rng(0)
    model = Sequential(Linear(2, 8, seed=1), Tanh(), Linear(8, 1, seed=2))
    optimizer = optimizer_cls(model.parameters(), **kwargs)
    x = Tensor(rng.uniform(-1, 1, size=(64, 2)))
    y = Tensor((x.data[:, 0:1] * x.data[:, 1:2] > 0).astype(np.float64))
    losses = []
    for _ in range(150):
        optimizer.zero_grad()
        loss = mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


@pytest.mark.parametrize(
    "optimizer_cls, kwargs",
    [
        (SGD, dict(lr=0.1)),
        (SGD, dict(lr=0.1, momentum=0.9)),
        (Adam, dict(lr=0.05)),
        (RMSProp, dict(lr=0.01)),
        (RMSProp, dict(lr=0.01, momentum=0.9)),
        (AdamW, dict(lr=0.05, weight_decay=0.01)),
    ],
)
def test_optimizer_decreases_loss(optimizer_cls, kwargs):
    losses = _fit_xor(optimizer_cls, **kwargs)
    assert losses[-1] < losses[0] * 0.5, f"{optimizer_cls.__name__} did not reduce loss enough: {losses[0]} -> {losses[-1]}"


@pytest.mark.parametrize("bad_kwargs", [dict(lr=0.0), dict(lr=-1.0), dict(lr=0.1, momentum=1.0)])
def test_sgd_rejects_invalid_hyperparameters(bad_kwargs):
    model = Sequential(Linear(2, 2, seed=1))
    with pytest.raises(ValueError):
        SGD(model.parameters(), **bad_kwargs)


def test_adamw_decays_weights_even_with_zero_gradient():
    model = Sequential(Linear(2, 2, seed=1))
    optimizer = AdamW(model.parameters(), lr=0.1, weight_decay=0.5)
    weight_before = model.layers[0].weight.data.copy()
    for p in model.parameters():
        p.zero_grad()
    optimizer.step()
    weight_after = model.layers[0].weight.data
    assert not np.allclose(weight_before, weight_after), "decoupled weight decay should shrink weights regardless of gradient"
    np.testing.assert_allclose(weight_after, weight_before * (1 - 0.1 * 0.5), atol=1e-10)


def test_clip_grad_norm_caps_total_norm():
    model = Sequential(Linear(2, 4, seed=1))
    for p in model.parameters():
        p.zero_grad()
        p.grad += 50.0
    total_before = clip_grad_norm_(model.parameters(), max_norm=1.0)
    total_after = float(np.sqrt(sum(np.sum(p.grad ** 2) for p in model.parameters())))
    assert total_before > 1.0
    assert total_after == pytest.approx(1.0, rel=1e-6)


def test_clip_grad_norm_is_noop_when_already_under_budget():
    model = Sequential(Linear(2, 4, seed=1))
    for p in model.parameters():
        p.zero_grad()
        p.grad += 0.001
    grads_before = [p.grad.copy() for p in model.parameters()]
    clip_grad_norm_(model.parameters(), max_norm=100.0)
    for before, p in zip(grads_before, model.parameters()):
        np.testing.assert_allclose(before, p.grad)


def test_step_lr_decays_on_schedule():
    model = Sequential(Linear(2, 2, seed=1))
    optimizer = SGD(model.parameters(), lr=1.0)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
    lrs = []
    for _ in range(6):
        scheduler.step()
        lrs.append(optimizer.lr)
    assert lrs == [1.0, 0.5, 0.5, 0.25, 0.25, 0.125]


def test_cosine_annealing_reaches_eta_min_and_holds():
    model = Sequential(Linear(2, 2, seed=1))
    optimizer = SGD(model.parameters(), lr=1.0)
    scheduler = CosineAnnealingLR(optimizer, t_max=10, eta_min=0.1)
    for _ in range(10):
        scheduler.step()
    assert optimizer.lr == pytest.approx(0.1, abs=1e-9)
    scheduler.step()
    assert optimizer.lr == pytest.approx(0.1, abs=1e-9)


def test_linear_warmup_ramps_then_holds():
    model = Sequential(Linear(2, 2, seed=1))
    optimizer = SGD(model.parameters(), lr=2.0)
    scheduler = LinearWarmupLR(optimizer, warmup_steps=4)
    lrs = []
    for _ in range(6):
        scheduler.step()
        lrs.append(optimizer.lr)
    assert lrs == [0.5, 1.0, 1.5, 2.0, 2.0, 2.0]
