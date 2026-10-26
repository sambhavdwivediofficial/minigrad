from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from minigrad import Tensor, Conv2d, MaxPool2d, ReLU, Flatten, Linear, Sequential, load_checkpoint

CLASS_NAMES = ["square", "circle", "cross"]


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
    parser = ArgumentParser(prog="predict_cnn_shapes")
    parser.add_argument("--true-class", choices=CLASS_NAMES, default=None, help="draw this shape instead of a random one")
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/cnn_shapes_model.npz"))
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}. Run examples/cnn_shapes_classification.py first.")
    return args


def main():
    args = parse_args()
    model = build_model()
    load_checkpoint(model, args.checkpoint)
    model.eval()

    rng = np.random.default_rng(args.seed)
    label_index = CLASS_NAMES.index(args.true_class) if args.true_class else int(rng.integers(0, len(DRAW_FUNCTIONS)))
    img = np.zeros((args.image_size, args.image_size))
    DRAW_FUNCTIONS[label_index](img, rng)
    img += rng.normal(0.0, args.noise, size=img.shape)

    logits = model(Tensor(img.reshape(1, 1, args.image_size, args.image_size)))
    probabilities = logits.softmax(axis=-1).data[0]
    predicted_index = int(logits.data[0].argmax())

    print(f"True shape drawn: {CLASS_NAMES[label_index]}")
    for class_name, probability in zip(CLASS_NAMES, probabilities):
        print(f"  {class_name:7s}: probability {probability:.4f}")
    print(f"Predicted shape: {CLASS_NAMES[predicted_index]}")
    print(f"Correct: {predicted_index == label_index}")


if __name__ == "__main__":
    main()
