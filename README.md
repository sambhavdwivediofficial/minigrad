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

## CNNs, RNNs, and attention

- **Convolutional:** `Conv2d`, `MaxPool2d`, `AvgPool2d`, `pad2d` — every one of them composed from already-tested `Tensor` ops (slicing, `reshape`, `stack`, 2-D `matmul`), with no convolution-specific backward code. See `docs/architecture.md`.
- **Recurrent:** `RNNCell`/`LSTMCell`/`GRUCell` plus `RNN`/`LSTM`/`GRU` sequence wrappers — real backpropagation-through-time, via a plain Python loop over timesteps.
- **Attention:** `scaled_dot_product_attention`, `MultiHeadSelfAttention`, `FeedForward`, `TransformerEncoderBlock`, and a minimal causal `TransformerLM`, plus `sinusoidal_positional_encoding`.
- **New examples:** `examples/cnn_shapes_classification.py` (image classification) and `examples/char_lstm_demo.py` (a trainable, text-generating character-level language model). See `docs/examples.md`.
- **New module:** `Embedding`, for anywhere you need a trainable lookup table (used by both the recurrent and attention layers).

## Batched matmul, second-order autodiff, and resumable training

```powershell
python .\examples\predict_regression.py 1.5
python .\examples\predict_moons.py 1.0 0.5
python .\examples\predict_spiral.py 0.3 0.2
python .\examples\predict_tabular.py 3 1800 15 5.0
python .\examples\predict_cnn_shapes.py --true-class circle

python .\examples\char_transformer_demo.py
python .\examples\resume_training_demo.py
```

- **True batched matmul:** `@` now supports any number of broadcasting leading batch dimensions (not just plain 2-D), the same way `np.matmul` does. `MultiHeadSelfAttention` no longer loops over `(batch, head)` pairs in Python — it is one batched matmul — and `Linear` now accepts input of any rank (`[batch, features]` or `[batch, seq, features]`) with no code changes, since a 2-D weight matrix broadcasts against a higher-rank input automatically. See `docs/architecture.md`.
- **Second-order autodiff:** `Tensor.backward(create_graph=True)` makes `.grad` a differentiable `Tensor`, so a second `.backward()` call differentiates through the first. `minigrad.grad`, `minigrad.hvp`, `minigrad.hessian`, and `minigrad.newton_step` wrap this in a small functional API — `newton_step` reaches the exact minimum of a quadratic in one step, which only happens if the computed Hessian is exactly right. Supported through every op except `__getitem__`'s own backward rule (so: `Linear`, `LayerNorm`, `BatchNorm1d`, `Dropout`, the regression/BCE losses, and a single RNN/LSTM/GRU cell call all support it; `Conv2d`, `Embedding`, `softmax_cross_entropy`, and the RNN/LSTM/GRU sequence wrappers do not yet). See `docs/architecture.md` and `LIMITATIONS.md`.
- **Resumable training:** `SGD`/`Adam`/`RMSProp`/`AdamW` now implement `state_dict()`/`load_state_dict()`, and `save_training_state`/`load_training_state` persist model and optimizer state together, so a resumed run is bit-for-bit identical to an uninterrupted one — see `examples/resume_training_demo.py`.
- **Predict scripts for every example:** `predict_regression.py`, `predict_moons.py`, `predict_spiral.py`, `predict_tabular.py`, and `predict_cnn_shapes.py` join `predict_xor.py`, each loading its matching checkpoint and running one prediction. See `docs/examples.md`.
- **Transformer text generation:** `char_transformer_demo.py` is a `TransformerLM`-based counterpart to `char_lstm_demo.py` — same next-character-prediction task, generated through attention instead of recurrence.

## Vectorized convolutions and two new diagnostic tools

```powershell
python .\examples\diagnostics_demo.py
```

- **`Tensor.unfold2d`, and a much faster `Conv2d`/`MaxPool2d`/`AvgPool2d`:** sliding windows are now extracted with `numpy.lib.stride_tricks.as_strided` (one vectorized NumPy call) instead of a Python loop over every output position. A 64×64 image's `Conv2d` forward+backward went from ~245ms to ~24ms in testing — roughly 10x, and the gap widens with image size — with byte-identical output and gradients to the loop-based version it replaced. See `docs/architecture.md`.
- **`explain_nan(tensor)`:** walks a computation graph forward and reports the exact operation where a NaN or Inf *first* appeared — not just that the final loss is NaN — along with a likely-cause hint (`log` of a non-positive value, division by zero, `exp` overflow, ...).
- **`explain_gradient(fn, x)`:** lists every operation on the path from `x` to `fn(x)`, each with a human-readable statement of its local derivative rule, then runs `gradcheck` and attaches the result as verification — the chain of rules shown is the literal sequence `Tensor.backward()` would execute, not a guess.
- Both tools exist because this engine's computation graph is plain, inspectable Python objects rather than a compiled trace — see `docs/architecture.md` for why that makes them straightforward here and unusually hard to build on top of a compiled framework.

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
