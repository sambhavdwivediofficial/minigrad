from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from minigrad import (
    Tensor, Linear, Sequential, Tanh,
    Adam, StepLR, huber_loss,
    TensorDataset, DataLoader, Trainer, EarlyStopping, r_squared,
    save_checkpoint,
)


def make_dataset(size, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, size=(size, 1))
    noise = rng.normal(0.0, 0.3, size=(size, 1))
    y = np.sin(x) * 2.0 + 0.5 * x + noise
    return x, y


def build_model():
    return Sequential(
        Linear(1, 32, seed=1),
        Tanh(),
        Linear(32, 32, seed=2),
        Tanh(),
        Linear(32, 1, seed=3),
    )


def parse_args():
    parser = ArgumentParser(prog="regression_demo")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--train-size", type=int, default=600)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/regression_model.npz"))
    parser.add_argument("--plot", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    x_train, y_train = make_dataset(args.train_size, args.seed)
    x_val, y_val = make_dataset(args.val_size, args.seed + 1)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True, seed=args.seed)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=256)

    model = build_model()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=80, gamma=0.5)

    def loss_fn(predictions, targets):
        return huber_loss(predictions, targets, delta=args.delta)

    trainer = Trainer(
        model, optimizer, loss_fn,
        metric_fns={"r2": r_squared},
        scheduler=scheduler,
    )
    history = trainer.fit(
        train_loader, val_loader,
        epochs=args.epochs,
        early_stopping=EarlyStopping(patience=40),
        checkpoint_path=args.checkpoint,
        verbose_every=50,
    )

    print(f"Final validation R^2: {history.val_metrics[-1]['r2']:.4f}")
    print(f"Saved model parameters to {args.checkpoint}")

    if args.plot is not None:
        from minigrad.visualization import plot_loss_curve
        plot_loss_curve(history, args.plot, title="regression_demo loss")
        print(f"Saved loss curve to {args.plot}")


if __name__ == "__main__":
    main()
