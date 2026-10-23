# MiniGrad: Deep Learning Framework

MiniGrad is an educational, NumPy-backed neural-network framework. It implements a small dynamic computation graph, reverse-mode automatic differentiation, neural-network layers, SGD/Adam optimizers, binary cross-entropy with logits, and model-parameter checkpoints.

**Scope:** this is a CPU-only learning project, not a replacement for PyTorch, TensorFlow, or JAX. Tensors use `float64`; matrix multiplication currently supports 2-D arrays only; many production features are intentionally omitted.

## Documentation

| Document | What's in it |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | How the autograd engine, `Module` system, and `Trainer` actually work internally |
| [`docs/api.md`](docs/api.md) | Full reference for every public class and function in the package |
| [`docs/examples.md`](docs/examples.md) | Every script in `examples/`, the command to run it, and its actual output (including the result images) |
| [`LIMITATIONS.md`](LIMITATIONS.md) | What this project intentionally does not do, and why |

## What this project teaches

- How a tensor stores values and gradients.
- How forward operations create a computation graph.
- How the chain rule is applied in reverse mode.
- How trainable parameters are updated by optimizers.
- How an MLP can be trained without PyTorch or TensorFlow.
- How to validate gradient code with unit tests.

## Requirements

- Python 3.10+
- NumPy
- pytest (development only)

## Setup

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Run tests

```bash
pytest -q
```
The tests cover core gradients, additional autodifferentiation operations, and checkpoint save/load behavior.

## Train

```powershell
python .\examples\train_xor.py
```

Training uses separate training, validation, and test datasets. It tracks validation loss, saves the best checkpoint, uses early stopping, and reports test metrics. The default checkpoint is `artifacts/xor_model.npz`.

Training options can be changed from the command line:

```powershell
python .\examples\train_xor.py --epochs 1500 --lr 0.02 --patience 150 --seed 12
python .\examples\train_xor.py --checkpoint artifacts\custom_xor.npz
```

## Predict one input pair

Run these commands after training:

```powershell
python .\examples\predict_xor.py 0.8 0.4
python .\examples\predict_xor.py -0.8 0.4
python .\examples\predict_xor.py 0.8 -0.4 --checkpoint artifacts\custom_xor.npz
```

The prediction command prints the estimated probability of class 1 and the predicted class. Same-sign input values are class 1; opposite-sign values are class 0.

## Evaluate a saved model

```powershell
python .\examples\evaluate_xor.py
python .\examples\evaluate_xor.py --samples 5000 --seed 42
python .\examples\evaluate_xor.py --threshold 0.6
python .\examples\evaluate_xor.py --checkpoint artifacts\custom_xor.npz
```

The evaluation command generates a new synthetic dataset and reports loss, accuracy, precision, recall, specificity, F1 score, and a confusion matrix. This is a synthetic XOR task, not evidence of performance on real-world data.

The script trains a small multilayer perceptron on a synthetic XOR-style classification task and writes model parameters to `artifacts/xor_model.npz`.

## Regression, classification, and tabular examples

```powershell
python -m pip install -e ".[dev,viz]"

python .\examples\regression_demo.py
python .\examples\regression_demo.py --plot artifacts\regression_loss.png

python .\examples\moons_spiral_classification.py --plot
python .\examples\moons_spiral_classification.py --task moons --plot
python .\examples\moons_spiral_classification.py --task spiral --spiral-classes 4 --plot

python .\examples\tabular_csv_demo.py
```

`regression_demo.py` fits a noisy nonlinear curve with Huber loss and reports R^2. `moons_spiral_classification.py` builds synthetic two-moons (binary) and spiral (multiclass) datasets, trains with the reusable `Trainer`, and with `--plot` saves a decision-boundary PNG under `artifacts/`. `tabular_csv_demo.py` synthesizes a small housing-price CSV, loads it with `CSVDataset`, standardizes it, and trains a `Linear + BatchNorm1d + ReLU` regressor with `AdamW`.

All three share the same reusable pieces now used across the project: `minigrad.data.{TensorDataset, CSVDataset, DataLoader}` for batching and `minigrad.engine.{Trainer, EarlyStopping}` for the train/validate/early-stop loop, instead of each script hand-rolling its own.

## What's new since v0.1.0

- **Tensor engine:** `reshape`, `transpose`/`.T`, axis-aware `sum`/`mean`, `max`/`argmax`, `exp`, `log`, `abs`, NumPy-style indexing (`__getitem__`), `log_softmax`/`softmax`, plus graph-level `where`/`concatenate`/`stack`.
- **Modules:** parameter/buffer separation, recursive `.train()`/`.eval()`, `Dropout`, `LayerNorm`, `BatchNorm1d`, `Flatten`.
- **Losses:** `mae_loss`, `huber_loss`, `softmax_cross_entropy`, `kl_divergence`.
- **Optimizers:** `RMSProp`, `AdamW` (decoupled weight decay), `clip_grad_norm_`, `StepLR`, `CosineAnnealingLR`, `LinearWarmupLR`.
- **Data/training:** `Dataset`/`TensorDataset`/`CSVDataset`/`DataLoader`, and a reusable `Trainer` + `EarlyStopping` (see `minigrad/engine.py`) that every example now shares.
- **Testing:** `gradcheck_multi` for multi-tensor ops, and a much larger test suite — shared-graph gradients, broadcasting sweeps, and a gradient check for every new op. See `LIMITATIONS.md` for what is intentionally still out of scope.

## Architecture

```text
Tensor / ndarray
      |
      v
Forward operations (+, *, /, matmul, tanh, ReLU, softplus, sum)
      |
      +---- record parents + local backward functions
      |
      v
Dynamic computation DAG
      |
      v
Tensor.backward() -> reverse topological traversal -> gradients
      |
      v
Module parameters -> SGD / Adam -> update parameter arrays
      |
      v
Next training iteration
```

## How to grow this into an advanced project

1. **Autograd correctness:** extend the included finite-difference gradient checker to every operation and test broadcasting edge cases.
2. **More tensor operations:** reshape, transpose, axis reductions, indexing/gather, concatenation, and batched matrix multiplication.
3. **Layers:** `Dropout`, `Embedding`, `LayerNorm`, `BatchNorm`, `Conv2D`, and pooling.
4. **Losses and optimizers:** softmax cross-entropy, momentum SGD, learning-rate schedulers, gradient clipping, weight decay, and optimizer state checkpoints.
5. **Training infrastructure:** minibatch `DataLoader`, train/eval modes, deterministic seeds, metric tracking, early stopping, and experiment configs.
6. **Performance:** profile operations, compare against NumPy and PyTorch, reduce graph memory, then consider a C++ backend.
7. **Advanced compiler work:** build a static graph representation and implement simple operation fusion or constant folding.
8. **GPU (optional):** add a GPU backend only after CPU correctness is well tested; this is a substantial separate milestone.

## What makes a credible portfolio submission?

- Passing unit tests and gradient checks.
- A benchmark report with exact hardware, shapes, timing method, and framework versions.
- Training curves and a held-out evaluation set.
- A written list of unsupported operations and known limitations.
- Reproducible commands and a clean package structure.

Do not claim MiniGrad is faster than major frameworks unless you have benchmarked it under controlled, comparable conditions. Its initial goal is correctness and educational clarity.
