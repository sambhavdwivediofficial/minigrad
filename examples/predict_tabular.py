from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from minigrad import Tensor, Linear, Sequential, ReLU, BatchNorm1d, load_checkpoint


def build_model():
    return Sequential(
        Linear(4, 32, seed=1), BatchNorm1d(32), ReLU(),
        Linear(32, 16, seed=2), ReLU(),
        Linear(16, 1, seed=3),
    )


def parse_args():
    parser = ArgumentParser(prog="predict_tabular")
    parser.add_argument("bedrooms", type=float)
    parser.add_argument("sqft", type=float)
    parser.add_argument("age", type=float)
    parser.add_argument("distance_to_center", type=float)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/tabular_model.npz"))
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}. Run examples/tabular_csv_demo.py first.")
    stats_path = args.checkpoint.with_name(args.checkpoint.stem + "_norm_stats.npz")
    if not stats_path.is_file():
        parser.error(f"normalization stats not found: {stats_path}. Run examples/tabular_csv_demo.py first.")
    return args, stats_path


def main():
    args, stats_path = parse_args()
    model = build_model()
    load_checkpoint(model, args.checkpoint)
    model.eval()

    with np.load(stats_path) as stats:
        x_mean, x_std = stats["x_mean"], stats["x_std"]
        y_mean, y_std = stats["y_mean"], stats["y_std"]

    raw = np.array([[args.bedrooms, args.sqft, args.age, args.distance_to_center]])
    standardized = (raw - x_mean) / x_std
    prediction_standardized = model(Tensor(standardized)).item()
    prediction_price = prediction_standardized * float(y_std) + float(y_mean)

    print(f"Input: bedrooms={args.bedrooms:.1f}, sqft={args.sqft:.1f}, age={args.age:.1f}, distance_to_center={args.distance_to_center:.2f}")
    print(f"Predicted price: {prediction_price:,.2f}")


if __name__ == "__main__":
    main()
