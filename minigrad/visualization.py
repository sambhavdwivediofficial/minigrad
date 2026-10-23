from __future__ import annotations

from pathlib import Path
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def _require_matplotlib():
    if not _HAS_MPL:
        raise ImportError(
            "minigrad.visualization needs matplotlib. Install it with:\n"
            '    python -m pip install -e ".[viz]"'
        )


def plot_loss_curve(history, path: str | Path, title: str = "Training loss") -> None:
    _require_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = range(1, len(history.train_loss) + 1)
    ax.plot(epochs, history.train_loss, label="train loss")
    if getattr(history, "val_loss", None):
        ax.plot(range(1, len(history.val_loss) + 1), history.val_loss, label="val loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_decision_boundary(
    model,
    x: np.ndarray,
    y: np.ndarray,
    path: str | Path,
    resolution: int = 200,
    title: str = "Decision boundary",
    multiclass: bool = False,
) -> None:
    _require_matplotlib()
    from .tensor import Tensor

    x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
    )
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    logits = model(Tensor(grid)).detach().data
    if multiclass:
        zz = logits.argmax(axis=-1).reshape(xx.shape)
    else:
        zz = (logits.reshape(-1) >= 0.0).astype(np.float64).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(xx, yy, zz, alpha=0.3, levels=np.unique(zz).size)
    labels = y.reshape(-1)
    scatter = ax.scatter(x[:, 0], x[:, 1], c=labels, edgecolors="k", s=18)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
