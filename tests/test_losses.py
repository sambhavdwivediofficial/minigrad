import numpy as np
import pytest
from minigrad import (
    Tensor, gradcheck,
    mse_loss, mae_loss, huber_loss, binary_cross_entropy_with_logits,
    softmax_cross_entropy, kl_divergence,
)


def test_mse_loss_gradcheck():
    rng = np.random.default_rng(0)
    pred = Tensor(rng.normal(size=(8,)), requires_grad=True)
    target = Tensor(rng.normal(size=(8,)))
    assert gradcheck(lambda p: mse_loss(p, target), pred).passed


def test_mae_loss_gradcheck():
    rng = np.random.default_rng(1)
    pred = Tensor(rng.normal(size=(8,)), requires_grad=True)
    target = Tensor(rng.normal(size=(8,)))
    assert gradcheck(lambda p: mae_loss(p, target), pred).passed


@pytest.mark.parametrize("delta", [0.5, 1.0, 2.0])
def test_huber_loss_gradcheck(delta):
    rng = np.random.default_rng(2)
    pred = Tensor(rng.normal(size=(10,)) * 3, requires_grad=True)
    target = Tensor(rng.normal(size=(10,)) * 3)
    assert gradcheck(lambda p: huber_loss(p, target, delta=delta), pred).passed


def test_huber_matches_quadratic_form_for_small_residuals():
    pred = Tensor([0.1], requires_grad=True)
    target = Tensor([0.0])
    huber = huber_loss(pred, target, delta=1.0).item()
    np.testing.assert_allclose(huber, 0.5 * 0.1 ** 2, atol=1e-10)


def test_huber_matches_linear_for_large_residuals():
    pred = Tensor([10.0], requires_grad=True)
    target = Tensor([0.0])
    huber = huber_loss(pred, target, delta=1.0).item()
    expected = 1.0 * (10.0 - 0.5 * 1.0)
    np.testing.assert_allclose(huber, expected, atol=1e-10)


def test_bce_with_logits_gradcheck():
    rng = np.random.default_rng(3)
    logits = Tensor(rng.normal(size=(8,)), requires_grad=True)
    target = Tensor((rng.random(8) > 0.5).astype(np.float64))
    assert gradcheck(lambda p: binary_cross_entropy_with_logits(p, target), logits).passed


def test_softmax_cross_entropy_gradcheck():
    rng = np.random.default_rng(4)
    logits = Tensor(rng.normal(size=(6, 4)), requires_grad=True)
    targets = rng.integers(0, 4, size=6)
    assert gradcheck(lambda p: softmax_cross_entropy(p, targets), logits).passed


def test_softmax_cross_entropy_matches_manual_computation():
    logits = Tensor([[2.0, 0.0, 0.0]])
    targets = np.array([0])
    loss = softmax_cross_entropy(logits, targets).item()
    probs = np.exp([2.0, 0.0, 0.0])
    probs /= probs.sum()
    expected = -np.log(probs[0])
    np.testing.assert_allclose(loss, expected, atol=1e-8)


def test_softmax_cross_entropy_rejects_bad_target_shape():
    logits = Tensor([[1.0, 2.0]])
    with pytest.raises(ValueError):
        softmax_cross_entropy(logits, np.array([[0]]))


def test_kl_divergence_gradcheck():
    rng = np.random.default_rng(5)
    logits = Tensor(rng.normal(size=(5, 4)), requires_grad=True)
    target_dist = rng.random((5, 4))
    target_dist /= target_dist.sum(axis=1, keepdims=True)
    assert gradcheck(lambda p: kl_divergence(p.log_softmax(axis=1), target_dist), logits).passed


def test_kl_divergence_is_zero_when_distributions_match():
    probs = np.array([[0.25, 0.25, 0.25, 0.25]])
    log_probs = Tensor(np.log(probs))
    kl = kl_divergence(log_probs, probs).item()
    np.testing.assert_allclose(kl, 0.0, atol=1e-10)
