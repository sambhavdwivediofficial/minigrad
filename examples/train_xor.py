"""Train a small MLP on XOR using only minigrad + NumPy."""
from pathlib import Path
import numpy as np
from minigrad import (
    Tensor, Sequential, Linear, Tanh, Adam,
    binary_cross_entropy_with_logits, save_checkpoint,
)


def main():
    rng = np.random.default_rng(7)
    x = rng.uniform(-1.0, 1.0, size=(1024, 2))
    # Same-sign quadrants are class 1; opposite-sign quadrants are class 0.
    y = ((x[:, 0] * x[:, 1]) > 0).astype(np.float64).reshape(-1, 1)

    model = Sequential(
        Linear(2, 16, seed=1), Tanh(),
        Linear(16, 16, seed=2), Tanh(),
        Linear(16, 1, seed=3),
    )
    optimizer = Adam(model.parameters(), lr=0.025)
    x_tensor = Tensor(x)
    y_tensor = Tensor(y)

    for epoch in range(1, 1001):
        optimizer.zero_grad()
        logits = model(x_tensor)
        loss = binary_cross_entropy_with_logits(logits, y_tensor)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 100 == 0:
            probabilities = logits.detach().sigmoid().data
            accuracy = np.mean((probabilities >= 0.5) == (y >= 0.5))
            print(f"epoch={epoch:4d} loss={loss.item():.5f} accuracy={accuracy:.3f}")

    path = Path("artifacts/xor_model.npz")
    save_checkpoint(model, path)
    print(f"Saved model parameters to {path}")


if __name__ == "__main__":
    main()
