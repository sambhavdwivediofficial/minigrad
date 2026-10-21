from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from minigrad import Tensor, binary_cross_entropy_with_logits, load_checkpoint
from xor_model import build_xor_model


def make_dataset(samples, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(samples, 2))
    y = ((x[:, 0] * x[:, 1]) > 0.0).astype(np.float64).reshape(-1, 1)
    return x, y


def calculate_metrics(probabilities, targets, threshold):
    predicted = (probabilities >= threshold).astype(np.int64)
    actual = targets.astype(np.int64)
    true_positive = int(np.sum((predicted == 1) & (actual == 1)))
    true_negative = int(np.sum((predicted == 0) & (actual == 0)))
    false_positive = int(np.sum((predicted == 1) & (actual == 0)))
    false_negative = int(np.sum((predicted == 0) & (actual == 1)))
    accuracy = (true_positive + true_negative) / max(actual.size, 1)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
    f1_denominator = precision + recall
    f1 = 2.0 * precision * recall / f1_denominator if f1_denominator else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
        "tp": true_positive,
    }


def parse_args():
    parser = ArgumentParser(prog="evaluate_xor")
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/xor_model.npz"))
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1")
    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold must be greater than 0 and less than 1")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}. Train the model first.")
    return args


def main():
    args = parse_args()
    x_test, y_test = make_dataset(args.samples, args.seed)
    model = build_xor_model()
    load_checkpoint(model, args.checkpoint)
    logits = model(Tensor(x_test))
    probabilities = logits.detach().sigmoid().data
    loss = binary_cross_entropy_with_logits(logits, Tensor(y_test)).item()
    metrics = calculate_metrics(probabilities, y_test, args.threshold)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples: {args.samples}")
    print(f"Threshold: {args.threshold:.3f}")
    print(f"Loss: {loss:.6f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"F1 score: {metrics['f1']:.4f}")
    print("Confusion matrix (rows=true, columns=predicted)")
    print("              Predicted 0  Predicted 1")
    print(f"Actual 0      {metrics['tn']:11d}  {metrics['fp']:11d}")
    print(f"Actual 1      {metrics['fn']:11d}  {metrics['tp']:11d}")


if __name__ == "__main__":
    main()
