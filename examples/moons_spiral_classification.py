from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from minigrad import (
    Linear, Sequential, Tanh, Dropout,
    Adam, binary_cross_entropy_with_logits, softmax_cross_entropy,
    TensorDataset, DataLoader, Trainer, EarlyStopping,
    binary_accuracy, multiclass_accuracy,
    save_checkpoint,
)


def make_moons(size, noise, seed):
    rng = np.random.default_rng(seed)
    n0 = size // 2
    n1 = size - n0
    theta0 = rng.uniform(0.0, np.pi, size=n0)
    theta1 = rng.uniform(0.0, np.pi, size=n1)
    x0 = np.stack([np.cos(theta0), np.sin(theta0)], axis=1)
    x1 = np.stack([1.0 - np.cos(theta1), 1.0 - np.sin(theta1) - 0.5], axis=1)
    x = np.concatenate([x0, x1], axis=0)
    x += rng.normal(0.0, noise, size=x.shape)
    y = np.concatenate([np.zeros(n0), np.ones(n1)]).reshape(-1, 1)
    return x, y


def make_spiral(size_per_class, num_classes, noise, seed):
    rng = np.random.default_rng(seed)
    x = np.zeros((size_per_class * num_classes, 2))
    y = np.zeros(size_per_class * num_classes, dtype=np.int64)
    for c in range(num_classes):
        idx = slice(c * size_per_class, (c + 1) * size_per_class)
        r = np.linspace(0.05, 1.0, size_per_class)
        t = np.linspace(c * 4.0, (c + 1) * 4.0, size_per_class) + rng.normal(0.0, noise, size_per_class)
        x[idx] = np.stack([r * np.sin(t), r * np.cos(t)], axis=1)
        y[idx] = c
    return x, y.reshape(-1, 1).astype(np.float64)


def run_moons(args):
    x, y = make_moons(args.moons_size, args.noise, args.seed)
    split = int(0.8 * len(x))
    train_loader = DataLoader(TensorDataset(x[:split], y[:split]), batch_size=32, shuffle=True, seed=args.seed)
    val_loader = DataLoader(TensorDataset(x[split:], y[split:]), batch_size=128)

    model = Sequential(Linear(2, 32, seed=1), Tanh(), Dropout(0.1, seed=2), Linear(32, 1, seed=3))
    trainer = Trainer(
        model, Adam(model.parameters(), lr=args.lr), binary_cross_entropy_with_logits,
        metric_fns={"acc": binary_accuracy},
    )
    history = trainer.fit(
        train_loader, val_loader, epochs=args.epochs,
        early_stopping=EarlyStopping(patience=30),
        checkpoint_path=args.checkpoint_dir / "moons_model.npz",
        verbose_every=50,
    )
    print(f"moons: final val acc={history.val_metrics[-1]['acc']:.4f}")

    if args.plot:
        from minigrad.visualization import plot_decision_boundary
        model.eval()
        plot_decision_boundary(
            model, x, y, args.checkpoint_dir / "moons_boundary.png", title="two-moons decision boundary"
        )
        print(f"Saved {args.checkpoint_dir / 'moons_boundary.png'}")


def run_spiral(args):
    x, y = make_spiral(args.spiral_per_class, args.spiral_classes, args.noise, args.seed)
    n = len(x)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n)
    x, y = x[order], y[order]
    split = int(0.8 * n)
    train_loader = DataLoader(TensorDataset(x[:split], y[:split]), batch_size=32, shuffle=True, seed=args.seed)
    val_loader = DataLoader(TensorDataset(x[split:], y[split:]), batch_size=128)

    model = Sequential(
        Linear(2, 64, seed=1), Tanh(),
        Linear(64, 64, seed=2), Tanh(),
        Linear(64, args.spiral_classes, seed=3),
    )

    def loss_fn(predictions, targets):
        return softmax_cross_entropy(predictions, targets.reshape(-1).astype(np.int64))

    def acc_fn(predictions, targets):
        return multiclass_accuracy(predictions, targets.reshape(-1).astype(np.int64))

    trainer = Trainer(
        model, Adam(model.parameters(), lr=args.lr), loss_fn,
        metric_fns={"acc": acc_fn},
    )
    history = trainer.fit(
        train_loader, val_loader, epochs=args.epochs,
        early_stopping=EarlyStopping(patience=30),
        checkpoint_path=args.checkpoint_dir / "spiral_model.npz",
        verbose_every=50,
    )
    print(f"spiral: final val acc={history.val_metrics[-1]['acc']:.4f}")

    if args.plot:
        from minigrad.visualization import plot_decision_boundary
        model.eval()
        plot_decision_boundary(
            model, x, y, args.checkpoint_dir / "spiral_boundary.png",
            title="spiral decision boundary", multiclass=True,
        )
        print(f"Saved {args.checkpoint_dir / 'spiral_boundary.png'}")


def parse_args():
    parser = ArgumentParser(prog="moons_spiral_classification")
    parser.add_argument("--task", choices=["moons", "spiral", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--noise", type=float, default=0.15)
    parser.add_argument("--moons-size", type=int, default=600)
    parser.add_argument("--spiral-per-class", type=int, default=200)
    parser.add_argument("--spiral-classes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.task in ("moons", "both"):
        run_moons(args)
    if args.task in ("spiral", "both"):
        run_spiral(args)


if __name__ == "__main__":
    main()
