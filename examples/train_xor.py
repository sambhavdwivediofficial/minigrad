from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from minigrad import Adam, Tensor, binary_cross_entropy_with_logits, save_checkpoint
from xor_model import build_xor_model


def make_dataset(size, rng):
    x = rng.uniform(-1.0, 1.0, size=(size, 2))
    y = ((x[:, 0] * x[:, 1]) > 0.0).astype(np.float64).reshape(-1, 1)
    return x, y


def evaluate(model, x, y):
    logits = model(Tensor(x))
    loss = binary_cross_entropy_with_logits(logits, Tensor(y))
    predictions = (logits.data >= 0.0).astype(np.float64)
    accuracy = float(np.mean(predictions == y))
    return float(loss.item()), accuracy


def parse_args():
    parser = ArgumentParser(prog="train_xor")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--train-size", type=int, default=1024)
    parser.add_argument("--validation-size", type=int, default=256)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/xor_model.npz"))
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if min(args.train_size, args.validation_size, args.test_size) < 1:
        parser.error("dataset sizes must be at least 1")
    if args.lr <= 0.0:
        parser.error("--lr must be greater than 0")
    if args.patience < 1:
        parser.error("--patience must be at least 1")
    if args.print_every < 1:
        parser.error("--print-every must be at least 1")
    return args


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    x_train, y_train = make_dataset(args.train_size, rng)
    x_validation, y_validation = make_dataset(args.validation_size, rng)
    x_test, y_test = make_dataset(args.test_size, rng)

    model = build_xor_model()
    optimizer = Adam(model.parameters(), lr=args.lr)
    train_x = Tensor(x_train)
    train_y = Tensor(y_train)
    best_validation_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        train_logits = model(train_x)
        train_loss = binary_cross_entropy_with_logits(train_logits, train_y)
        train_loss.backward()
        optimizer.step()

        validation_loss, validation_accuracy = evaluate(model, x_validation, y_validation)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = model.state_dict()
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            train_accuracy = float(np.mean((train_logits.data >= 0.0) == y_train))
            print(
                f"epoch={epoch:4d} "
                f"train_loss={train_loss.item():.5f} "
                f"train_accuracy={train_accuracy:.3f} "
                f"validation_loss={validation_loss:.5f} "
                f"validation_accuracy={validation_accuracy:.3f}"
            )

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("training did not produce a valid model state")

    model.load_state_dict(best_state)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, args.checkpoint)
    test_loss, test_accuracy = evaluate(model, x_test, y_test)

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_validation_loss:.5f}")
    print(f"Test loss: {test_loss:.5f}")
    print(f"Test accuracy: {test_accuracy:.3f}")
    print(f"Saved model parameters to {args.checkpoint}")


if __name__ == "__main__":
    main()
