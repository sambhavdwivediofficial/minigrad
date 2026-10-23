import numpy as np
import pytest
from minigrad import (
    Tensor, gradcheck, Linear, Sequential, Tanh, ReLU, Flatten,
    Dropout, LayerNorm, BatchNorm1d, save_checkpoint, load_checkpoint,
)


def build_model():
    return Sequential(
        Linear(3, 5, seed=1),
        BatchNorm1d(5),
        ReLU(),
        Dropout(p=0.3, seed=2),
        Linear(5, 2, seed=3),
    )


def test_train_eval_propagates_recursively():
    model = build_model()
    assert model.training and model.layers[1].training and model.layers[3].training
    model.eval()
    assert not model.training
    assert not model.layers[1].training, "BatchNorm1d did not switch to eval"
    assert not model.layers[3].training, "Dropout did not switch to eval"
    model.train()
    assert model.training and model.layers[1].training and model.layers[3].training


def test_flatten_shape():
    x = Tensor(np.random.randn(4, 3, 2))
    out = Flatten()(x)
    assert out.shape == (4, 6)


def test_layernorm_normalizes_each_row():
    ln = LayerNorm(4)
    x = Tensor(np.random.randn(6, 4) * 5 + 3)
    out = ln(x)
    row_mean = out.data.mean(axis=-1)
    row_var = out.data.var(axis=-1)
    np.testing.assert_allclose(row_mean, np.zeros(6), atol=1e-6)
    np.testing.assert_allclose(row_var, np.ones(6), atol=1e-3)


def test_layernorm_gradcheck():
    ln = LayerNorm(4)
    x = Tensor(np.random.randn(5, 4), requires_grad=True)
    assert gradcheck(lambda t: (ln(t) ** 2).sum(), x).passed


def test_batchnorm_train_mode_normalizes_batch():
    bn = BatchNorm1d(3)
    bn.train()
    x = Tensor(np.random.randn(50, 3) * 4 + 2)
    out = bn(x)
    np.testing.assert_allclose(out.data.mean(axis=0), np.zeros(3), atol=1e-6)
    np.testing.assert_allclose(out.data.var(axis=0), np.ones(3), atol=1e-2)


def test_batchnorm_updates_running_stats_only_in_train_mode():
    bn = BatchNorm1d(3)
    before = bn.buffer("running_mean").copy()
    bn.train()
    bn(Tensor(np.random.randn(20, 3) * 5 + 10))
    after_train = bn.buffer("running_mean").copy()
    assert not np.allclose(before, after_train)

    bn.eval()
    bn(Tensor(np.random.randn(20, 3) * 5 + 10))
    after_eval = bn.buffer("running_mean").copy()
    np.testing.assert_allclose(after_train, after_eval, err_msg="eval-mode forward must not update running stats")


def test_batchnorm_gradcheck_both_modes():
    bn = BatchNorm1d(3)
    bn.train()
    x = Tensor(np.random.randn(8, 3), requires_grad=True)
    assert gradcheck(lambda t: (bn(t) ** 2).sum(), x).passed
    bn.eval()
    x2 = Tensor(np.random.randn(5, 3), requires_grad=True)
    assert gradcheck(lambda t: (bn(t) ** 2).sum(), x2).passed


def test_dropout_identity_when_p_zero():
    drop = Dropout(p=0.0)
    x = Tensor(np.random.randn(4, 4), requires_grad=True)
    assert gradcheck(lambda t: (drop(t) ** 2).sum(), x).passed


def test_dropout_identity_in_eval_mode():
    drop = Dropout(p=0.9, seed=0)
    drop.eval()
    x = Tensor(np.ones((10, 10)))
    out = drop(x)
    np.testing.assert_allclose(out.data, x.data)


def test_dropout_zeroes_and_rescales_in_train_mode():
    drop = Dropout(p=0.5, seed=0)
    drop.train()
    x = Tensor(np.ones((2000,)))
    out = drop(x).data
    survivors = out[out != 0.0]
    np.testing.assert_allclose(survivors, np.full_like(survivors, 2.0))
    keep_fraction = (out != 0.0).mean()
    assert 0.4 < keep_fraction < 0.6


def test_dropout_rejects_invalid_probability():
    with pytest.raises(ValueError):
        Dropout(p=1.0)
    with pytest.raises(ValueError):
        Dropout(p=-0.1)


def test_checkpoint_round_trips_parameters_and_buffers(tmp_path):
    model = build_model()
    model.train()
    x = Tensor(np.random.randn(16, 3) * 3 + 1)
    for _ in range(5):
        model(x)

    model.eval()
    expected_output = model(x).data.copy()
    path = tmp_path / "model.npz"
    save_checkpoint(model, path)

    restored = build_model()
    load_checkpoint(restored, path)
    restored.eval()

    for name, value in model.state_dict().items():
        np.testing.assert_allclose(restored.state_dict()[name], value)
    np.testing.assert_allclose(restored(x).data, expected_output)


def test_num_parameters_counts_trainable_elements_only():
    model = Sequential(Linear(3, 5, seed=1), BatchNorm1d(5), Linear(5, 2, seed=2))
    assert model.num_parameters() == 20 + 10 + 12
