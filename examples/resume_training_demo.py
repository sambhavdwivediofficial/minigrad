from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from minigrad import (
    Tensor, Linear, Sequential, Tanh,
    Adam, binary_cross_entropy_with_logits,
    save_training_state, load_training_state,
)


def make_dataset(size, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(size, 2))
    y = ((x[:, 0] * x[:, 1]) > 0.0).astype(np.float64).reshape(-1, 1)
    return x, y


def build_model():
    return Sequential(Linear(2, 16, seed=1), Tanh(), Linear(16, 16, seed=2), Tanh(), Linear(16, 1, seed=3))


def run_segment(model, optimizer, x, y, start_epoch, num_epochs, print_every=20):
    x_t, y_t = Tensor(x), Tensor(y)
    loss_value = None
    for local_step in range(num_epochs):
        epoch = start_epoch + local_step + 1
        optimizer.zero_grad()
        logits = model(x_t)
        loss = binary_cross_entropy_with_logits(logits, y_t)
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        if epoch == 1 or epoch % print_every == 0:
            print(f"epoch={epoch:4d} loss={loss_value:.6f}")
    return epoch, loss_value


def parse_args():
    parser = ArgumentParser(prog="resume_training_demo")
    parser.add_argument("--train-size", type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--first-segment-epochs", type=int, default=150)
    parser.add_argument("--second-segment-epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/resume_demo_state.npz"))
    return parser.parse_args()


def main():
    args = parse_args()
    x, y = make_dataset(args.train_size, args.seed)

    print("=== Segment 1: training from scratch ===")
    model = build_model()
    optimizer = Adam(model.parameters(), lr=args.lr)
    epoch, loss_value = run_segment(model, optimizer, x, y, start_epoch=0, num_epochs=args.first_segment_epochs)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_training_state(model, optimizer, args.checkpoint, epoch=epoch, loss=loss_value)
    print(f"Saved full training state (model + optimizer + epoch) to {args.checkpoint}")
    print("Process would normally exit here; a fresh process now loads the checkpoint back.")

    print()
    print("=== Segment 2: a FRESH model/optimizer, resumed from the checkpoint ===")
    resumed_model = build_model()
    resumed_optimizer = Adam(resumed_model.parameters(), lr=args.lr)
    metadata = load_training_state(resumed_model, resumed_optimizer, args.checkpoint)
    resumed_epoch = int(metadata["epoch"])
    print(f"Restored epoch={resumed_epoch}, loss-at-save={float(metadata['loss']):.6f}")
    run_segment(resumed_model, resumed_optimizer, x, y, start_epoch=resumed_epoch, num_epochs=args.second_segment_epochs)

    print()
    print("=== Control: the SAME total number of epochs, trained uninterrupted ===")
    control_model = build_model()
    control_optimizer = Adam(control_model.parameters(), lr=args.lr)
    run_segment(
        control_model, control_optimizer, x, y, start_epoch=0,
        num_epochs=args.first_segment_epochs + args.second_segment_epochs, print_every=10**9,
    )
    control_loss = binary_cross_entropy_with_logits(control_model(Tensor(x)), Tensor(y)).item()
    resumed_loss = binary_cross_entropy_with_logits(resumed_model(Tensor(x)), Tensor(y)).item()
    print(f"Uninterrupted final loss: {control_loss:.8f}")
    print(f"Resumed final loss:       {resumed_loss:.8f}")
    print(f"Match within tolerance:   {abs(control_loss - resumed_loss) < 1e-6}")


if __name__ == "__main__":
    main()
