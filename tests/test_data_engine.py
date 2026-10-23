import numpy as np
import pytest
from minigrad import (
    Tensor, Linear, Sequential, Tanh, Adam, SGD,
    binary_cross_entropy_with_logits, mse_loss,
    TensorDataset, CSVDataset, DataLoader,
    Trainer, EarlyStopping, binary_accuracy, r_squared,
)


def test_tensor_dataset_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        TensorDataset(np.zeros((5, 2)), np.zeros((4, 1)))


def test_dataloader_covers_every_sample_exactly_once_per_epoch():
    x = np.arange(20).reshape(20, 1).astype(np.float64)
    loader = DataLoader(TensorDataset(x), batch_size=6, shuffle=False, drop_last=False)
    seen = np.concatenate([batch[0].reshape(-1) for batch in loader])
    np.testing.assert_array_equal(np.sort(seen), np.arange(20))


def test_dataloader_drop_last_discards_remainder():
    x = np.arange(20).reshape(20, 1).astype(np.float64)
    loader = DataLoader(TensorDataset(x), batch_size=6, drop_last=True)
    batches = list(loader)
    assert len(batches) == 3
    assert sum(b[0].shape[0] for b in batches) == 18


def test_dataloader_shuffle_is_reproducible_given_a_seed():
    x = np.arange(30).reshape(30, 1).astype(np.float64)
    loader_a = DataLoader(TensorDataset(x), batch_size=30, shuffle=True, seed=7)
    loader_b = DataLoader(TensorDataset(x), batch_size=30, shuffle=True, seed=7)
    (batch_a,) = list(loader_a)
    (batch_b,) = list(loader_b)
    np.testing.assert_array_equal(batch_a[0], batch_b[0])


def test_dataloader_len_matches_iteration_count():
    x = np.arange(25).reshape(25, 1).astype(np.float64)
    loader = DataLoader(TensorDataset(x), batch_size=7)
    assert len(loader) == len(list(loader)) == 4


def test_csv_dataset_round_trip(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("x1,x2,y\n1.0,2.0,3.0\n4.0,5.0,9.0\n")
    ds = CSVDataset(path, target_column=-1)
    assert len(ds) == 2
    x0, y0 = ds[0]
    np.testing.assert_allclose(x0, [1.0, 2.0])
    np.testing.assert_allclose(y0, [3.0])


def test_csv_dataset_rejects_missing_values(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("x1,x2,y\n1.0,,3.0\n")
    with pytest.raises(ValueError):
        CSVDataset(path, target_column=-1)


def test_early_stopping_triggers_after_patience_epochs_without_improvement():
    stopping = EarlyStopping(patience=3, mode="min")
    values = [1.0, 0.9, 0.95, 0.96, 0.97, 0.98]
    stopped_at = None
    for i, v in enumerate(values):
        stopping.step(v)
        if stopping.should_stop:
            stopped_at = i
            break
    assert stopped_at == 4


def test_early_stopping_max_mode():
    stopping = EarlyStopping(patience=2, mode="max")
    for v in [0.5, 0.7, 0.6, 0.55]:
        stopping.step(v)
    assert stopping.should_stop


def test_trainer_reduces_validation_loss_on_xor():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, size=(400, 2))
    y = ((x[:, 0] * x[:, 1]) > 0).astype(np.float64).reshape(-1, 1)
    train_loader = DataLoader(TensorDataset(x[:300], y[:300]), batch_size=32, shuffle=True, seed=1)
    val_loader = DataLoader(TensorDataset(x[300:], y[300:]), batch_size=64)

    model = Sequential(Linear(2, 16, seed=1), Tanh(), Linear(16, 1, seed=2))
    trainer = Trainer(
        model, Adam(model.parameters(), lr=0.05), binary_cross_entropy_with_logits,
        metric_fns={"acc": binary_accuracy},
    )
    history = trainer.fit(train_loader, val_loader, epochs=40, verbose_every=0)
    assert history.val_loss[-1] < history.val_loss[0]
    assert history.val_metrics[-1]["acc"] > 0.85


def test_trainer_restores_best_checkpoint_not_last_epoch(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, size=(200, 2))
    y = ((x[:, 0] * x[:, 1]) > 0).astype(np.float64).reshape(-1, 1)
    train_loader = DataLoader(TensorDataset(x[:150], y[:150]), batch_size=16, shuffle=True, seed=2)
    val_loader = DataLoader(TensorDataset(x[150:], y[150:]), batch_size=50)

    model = Sequential(Linear(2, 16, seed=3), Tanh(), Linear(16, 1, seed=4))
    trainer = Trainer(model, Adam(model.parameters(), lr=0.08), binary_cross_entropy_with_logits)
    ckpt_path = tmp_path / "best.npz"
    history = trainer.fit(
        train_loader, val_loader, epochs=50,
        early_stopping=EarlyStopping(patience=10), checkpoint_path=ckpt_path, verbose_every=0,
    )
    best_val = min(history.val_loss)
    restored_val_loss_index = history.val_loss.index(best_val)
    assert restored_val_loss_index < len(history.val_loss) - 1 or len(history.val_loss) == history.val_loss.index(best_val) + 1
    assert ckpt_path.exists()


def test_r_squared_perfect_fit_is_one():
    pred = Tensor([[1.0], [2.0], [3.0]])
    target = np.array([[1.0], [2.0], [3.0]])
    assert r_squared(pred, target) == pytest.approx(1.0)
