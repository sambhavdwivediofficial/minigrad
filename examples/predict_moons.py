from argparse import ArgumentParser
from pathlib import Path
from minigrad import Tensor, Linear, Sequential, Tanh, Dropout, load_checkpoint


def build_model():
    return Sequential(Linear(2, 32, seed=1), Tanh(), Dropout(0.1, seed=2), Linear(32, 1, seed=3))


def parse_args():
    parser = ArgumentParser(prog="predict_moons")
    parser.add_argument("x1", type=float)
    parser.add_argument("x2", type=float)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/moons_model.npz"))
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}. Run examples/moons_spiral_classification.py first.")
    return args


def main():
    args = parse_args()
    model = build_model()
    load_checkpoint(model, args.checkpoint)
    model.eval()
    logit = model(Tensor([[args.x1, args.x2]]))
    probability = logit.sigmoid().item()
    predicted_class = int(logit.item() >= 0.0)
    print(f"Input: x1={args.x1:.4f}, x2={args.x2:.4f}")
    print(f"Probability of class 1: {probability:.4f}")
    print(f"Predicted class: {predicted_class}")


if __name__ == "__main__":
    main()
