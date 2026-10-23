from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
import numpy as np

from .tensor import Tensor
from .nn import Module
from .optim import Optimizer, clip_grad_norm_
from .checkpoint import save_checkpoint


@dataclass
class History:
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    train_metrics: list = field(default_factory=list)
    val_metrics: list = field(default_factory=list)


class EarlyStopping:

    def __init__(self, patience: int = 10, mode: str = "min", min_delta: float = 0.0):
        if patience < 1:
            raise ValueError("patience must be >= 1")
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = patience
        self.mode = mode
        self.min_delta = float(min_delta)
        self.best: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        if self.best is None:
            improved = True
        elif self.mode == "min":
            improved = value < self.best - self.min_delta
        else:
            improved = value > self.best + self.min_delta
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


def _to_array(value) -> np.ndarray:
    return value.data if isinstance(value, Tensor) else np.asarray(value)


def binary_accuracy(predictions: Tensor, targets, threshold: float = 0.0) -> float:
    predicted = (predictions.data >= threshold).astype(np.float64)
    return float(np.mean(predicted == _to_array(targets)))


def multiclass_accuracy(predictions: Tensor, targets) -> float:
    targets = _to_array(targets)
    predicted = predictions.data.argmax(axis=-1)
    return float(np.mean(predicted == targets))


def r_squared(predictions: Tensor, targets) -> float:
    y = _to_array(targets).reshape(predictions.data.shape)
    y_hat = predictions.data
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


class Trainer:

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        loss_fn: Callable,
        metric_fns: Optional[dict] = None,
        scheduler=None,
        grad_clip: Optional[float] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric_fns = metric_fns or {}
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.history = History()

    def _run_epoch(self, loader, train: bool):
        self.model.train() if train else self.model.eval()
        total_loss = 0.0
        total_count = 0
        metric_totals = {name: 0.0 for name in self.metric_fns}

        for x_batch, y_batch in loader:
            x = Tensor(x_batch)
            batch_size = x_batch.shape[0]
            if train:
                self.optimizer.zero_grad()
            predictions = self.model(x)
            loss = self.loss_fn(predictions, y_batch)
            if train:
                loss.backward()
                if self.grad_clip is not None:
                    clip_grad_norm_(self.optimizer.parameters, self.grad_clip)
                self.optimizer.step()

            total_loss += loss.item() * batch_size
            total_count += batch_size
            detached_predictions = predictions.detach()
            for name, fn in self.metric_fns.items():
                metric_totals[name] += fn(detached_predictions, y_batch) * batch_size

        avg_loss = total_loss / max(total_count, 1)
        avg_metrics = {name: value / max(total_count, 1) for name, value in metric_totals.items()}
        return avg_loss, avg_metrics

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 100,
        early_stopping: Optional[EarlyStopping] = None,
        checkpoint_path: Optional[str | Path] = None,
        verbose_every: int = 1,
    ) -> History:
        best_state = None
        best_val = float("inf")

        for epoch in range(1, epochs + 1):
            train_loss, train_metrics = self._run_epoch(train_loader, train=True)
            self.history.train_loss.append(train_loss)
            self.history.train_metrics.append(train_metrics)

            val_loss = None
            val_metrics = {}
            if val_loader is not None:
                val_loss, val_metrics = self._run_epoch(val_loader, train=False)
                self.history.val_loss.append(val_loss)
                self.history.val_metrics.append(val_metrics)
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = self.model.state_dict()
                if early_stopping is not None:
                    early_stopping.step(val_loss)

            if self.scheduler is not None:
                self.scheduler.step()

            if verbose_every and (epoch == 1 or epoch % verbose_every == 0 or epoch == epochs):
                msg = f"epoch={epoch:4d} train_loss={train_loss:.5f}"
                for name, value in train_metrics.items():
                    msg += f" train_{name}={value:.3f}"
                if val_loss is not None:
                    msg += f" val_loss={val_loss:.5f}"
                    for name, value in val_metrics.items():
                        msg += f" val_{name}={value:.3f}"
                print(msg)

            if early_stopping is not None and early_stopping.should_stop:
                print(f"Early stopping at epoch {epoch}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            if checkpoint_path is not None:
                save_checkpoint(self.model, checkpoint_path)
        elif checkpoint_path is not None:
            save_checkpoint(self.model, checkpoint_path)

        return self.history
