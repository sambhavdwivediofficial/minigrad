from argparse import ArgumentParser
from pathlib import Path
from minigrad import Tensor, Linear, Sequential, Tanh, load_checkpoint


def build_model(num_classes: int):
    return Sequential(
        Linear(2, 64, seed=1), Tanh(),
        Linear(64, 64, seed=2), Tanh(),
        Linear(64, num_classes, seed=3),
    )


def parse_args():
    parser = ArgumentParser(prog="predict_spiral")
    parser.add_argument("x1", type=float)
    parser.add_argument("x2", type=float)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/spiral_model.npz"))
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}. Run examples/moons_spiral_classification.py first.")
    return args


def main():
    args = parse_args()
    model = build_model(args.num_classes)
    load_checkpoint(model, args.checkpoint)
    model.eval()
    logits = model(Tensor([[args.x1, args.x2]]))
    probabilities = logits.softmax(axis=-1).data[0]
    predicted_class = int(logits.data[0].argmax())
    print(f"Input: x1={args.x1:.4f}, x2={args.x2:.4f}")
    for class_idx, probability in enumerate(probabilities):
        print(f"  class {class_idx}: probability {probability:.4f}")
    print(f"Predicted class: {predicted_class}")


if __name__ == "__main__":
    main()
