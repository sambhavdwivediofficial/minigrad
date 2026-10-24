from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from minigrad import (
    Tensor, Conv2d, MaxPool2d, ReLU, Flatten, Linear, Sequential,
    Adam, softmax_cross_entropy,
    TensorDataset, DataLoader, Trainer, EarlyStopping, multiclass_accuracy,
)


def draw_square(img, rng):
    size = rng.integers(6, 11)
    top = rng.integers(0, img.shape[0] - size)
    left = rng.integers(0, img.shape[1] - size)
    img[top : top + size, left : left + size] = 1.0


def draw_cross(img, rng):
    thickness = rng.integers(2, 4)
    center_r = rng.integers(6, img.shape[0] - 6)
    center_c = rng.integers(6, img.shape[1] - 6)
    half = rng.integers(5, 8)
    img[center_r - thickness : center_r + thickness, center_c - half : center_c + half] = 1.0
    img[center_r - half : center_r + half, center_c - thickness : center_c + thickness] = 1.0


def draw_circle(img, rng):
    radius = rng.integers(4, 7)
    center_r = rng.integers(radius + 1, img.shape[0] - radius - 1)
    center_c = rng.integers(radius + 1, img.shape[1] - radius - 1)
    yy, xx = np.ogrid[: img.shape[0], : img.shape[1]]
    mask = (yy - center_r) ** 2 + (xx - center_c) ** 2 <= radius ** 2
    img[mask] = 1.0


DRAW_FUNCTIONS = [draw_square, draw_circle, draw_cross]
CLASS_NAMES = ["square", "circle", "cross"]


def make_dataset(size, image_size, noise, seed):
    rng = np.random.default_rng(seed)
    images = np.zeros((size, 1, image_size, image_size), dtype=np.float64)
    labels = np.zeros(size, dtype=np.int64)
    for i in range(size):
        label = rng.integers(0, len(DRAW_FUNCTIONS))
        img = np.zeros((image_size, image_size))
        DRAW_FUNCTIONS[label](img, rng)
        img += rng.normal(0.0, noise, size=img.shape)
        images[i, 0] = img
        labels[i] = label
    return images, labels


def build_model():
    return Sequential(
        Conv2d(1, 8, kernel_size=3, padding=1, seed=1),
        ReLU(),
        MaxPool2d(2),
        Conv2d(8, 16, kernel_size=3, padding=1, seed=2),
        ReLU(),
        MaxPool2d(2),
        Flatten(),
        Linear(16 * 4 * 4, 32, seed=3),
        ReLU(),
        Linear(32, len(CLASS_NAMES), seed=4),
    )


def parse_args():
    parser = ArgumentParser(prog="cnn_shapes_classification")
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--train-size", type=int, default=1200)
    parser.add_argument("--val-size", type=int, default=300)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/cnn_shapes_model.npz"))
    return parser.parse_args()


def main():
    args = parse_args()
    x_train, y_train = make_dataset(args.train_size, args.image_size, args.noise, args.seed)
    x_val, y_val = make_dataset(args.val_size, args.image_size, args.noise, args.seed + 1)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True, seed=args.seed)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=64)

    model = build_model()
    optimizer = Adam(model.parameters(), lr=args.lr)

    def loss_fn(predictions, targets):
        return softmax_cross_entropy(predictions, targets.astype(np.int64))

    def acc_fn(predictions, targets):
        return multiclass_accuracy(predictions, targets.astype(np.int64))

    trainer = Trainer(model, optimizer, loss_fn, metric_fns={"acc": acc_fn})
    history = trainer.fit(
        train_loader, val_loader,
        epochs=args.epochs,
        early_stopping=EarlyStopping(patience=8),
        checkpoint_path=args.checkpoint,
        verbose_every=5,
    )

    print(f"Classes: {CLASS_NAMES}")
    print(f"Final validation accuracy: {history.val_metrics[-1]['acc']:.4f}")
    print(f"Saved model parameters to {args.checkpoint}")


if __name__ == "__main__":
    main()
