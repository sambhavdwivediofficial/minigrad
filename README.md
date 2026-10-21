# MiniGrad: Build a Deep Learning Framework

MiniGrad is an educational, NumPy-backed neural-network framework. It implements a small dynamic computation graph, reverse-mode automatic differentiation, neural-network layers, SGD/Adam optimizers, binary cross-entropy with logits, and model-parameter checkpoints.

**Scope:** this is a CPU-only learning project, not a replacement for PyTorch, TensorFlow, or JAX. Tensors use `float64`; matrix multiplication currently supports 2-D arrays only; many production features are intentionally omitted.

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

## Train the XOR example

```bash
python examples/train_xor.py
```
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
