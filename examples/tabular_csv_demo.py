from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from minigrad import (
    Linear, Sequential, ReLU, BatchNorm1d,
    AdamW, mse_loss,
    CSVDataset, DataLoader, Trainer, EarlyStopping, r_squared,
)


def synthesize_csv(path, size, seed):
    rng = np.random.default_rng(seed)
    bedrooms = rng.integers(1, 6, size=size).astype(np.float64)
    sqft = rng.uniform(400, 3500, size=size)
    age = rng.uniform(0, 80, size=size)
    distance_to_center = rng.uniform(0.5, 25.0, size=size)
    noise = rng.normal(0.0, 15000.0, size=size)
    price = (
        50000
        + 120 * sqft
        + 8000 * bedrooms
        - 600 * age
        - 3000 * distance_to_center
        + noise
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("bedrooms,sqft,age,distance_to_center,price\n")
        for row in zip(bedrooms, sqft, age, distance_to_center, price):
            f.write(",".join(f"{v:.4f}" for v in row) + "\n")


def parse_args():
    parser = ArgumentParser(prog="tabular_csv_demo")
    parser.add_argument("--csv-path", type=Path, default=Path("artifacts/housing.csv"))
    parser.add_argument("--size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/tabular_model.npz"))
    return parser.parse_args()


def standardize(x, mean, std):
    return (x - mean) / std


def main():
    args = parse_args()
    synthesize_csv(args.csv_path, args.size, args.seed)

    dataset = CSVDataset(args.csv_path, target_column=-1)
    n = len(dataset)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = order[:split], order[split:]

    x_mean = dataset.x[train_idx].mean(axis=0, keepdims=True)
    x_std = dataset.x[train_idx].std(axis=0, keepdims=True) + 1e-8
    y_mean = dataset.y[train_idx].mean()
    y_std = dataset.y[train_idx].std() + 1e-8

    x_train = standardize(dataset.x[train_idx], x_mean, x_std)
    y_train = standardize(dataset.y[train_idx], y_mean, y_std)
    x_val = standardize(dataset.x[val_idx], x_mean, x_std)
    y_val = standardize(dataset.y[val_idx], y_mean, y_std)

    from minigrad import TensorDataset
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True, seed=args.seed)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=256)

    model = Sequential(
        Linear(x_train.shape[1], 32, seed=1), BatchNorm1d(32), ReLU(),
        Linear(32, 16, seed=2), ReLU(),
        Linear(16, 1, seed=3),
    )
    trainer = Trainer(
        model, AdamW(model.parameters(), lr=args.lr, weight_decay=0.001), mse_loss,
        metric_fns={"r2": r_squared},
    )
    history = trainer.fit(
        train_loader, val_loader, epochs=args.epochs,
        early_stopping=EarlyStopping(patience=25),
        checkpoint_path=args.checkpoint,
        verbose_every=30,
    )
    print(f"tabular_csv_demo: final val R^2 (standardized target)={history.val_metrics[-1]['r2']:.4f}")
    print(f"Saved model parameters to {args.checkpoint}")


if __name__ == "__main__":
    main()
