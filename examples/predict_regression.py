from argparse import ArgumentParser
from pathlib import Path
from minigrad import Tensor, Linear, Sequential, Tanh, load_checkpoint


def build_model():
    return Sequential(
        Linear(1, 32, seed=1),
        Tanh(),
        Linear(32, 32, seed=2),
        Tanh(),
        Linear(32, 1, seed=3),
    )


def parse_args():
    parser = ArgumentParser(prog="predict_regression")
    parser.add_argument("x", type=float)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/regression_model.npz"))
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}. Run examples/regression_demo.py first.")
    return args


def main():
    args = parse_args()
    model = build_model()
    load_checkpoint(model, args.checkpoint)
    model.eval()
    prediction = model(Tensor([[args.x]])).item()
    print(f"Input: x={args.x:.4f}")
    print(f"Predicted y: {prediction:.4f}")


if __name__ == "__main__":
    main()
