from argparse import ArgumentParser
from pathlib import Path
from minigrad import Tensor, load_checkpoint
from xor_model import build_xor_model


def parse_args():
    parser = ArgumentParser(prog="predict_xor")
    parser.add_argument("x1", type=float)
    parser.add_argument("x2", type=float)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/xor_model.npz"))
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}. Train the model first.")
    return args


def main():
    args = parse_args()
    model = build_xor_model()
    load_checkpoint(model, args.checkpoint)
    output = model(Tensor([[args.x1, args.x2]]))
    logit = float(output.data.item())
    probability = float(output.detach().sigmoid().data.item())
    predicted_class = int(logit >= 0.0)

    print(f"Input: x1={args.x1:.4f}, x2={args.x2:.4f}")
    print(f"Probability of class 1: {probability:.4f}")
    print(f"Predicted class: {predicted_class}")


if __name__ == "__main__":
    main()
