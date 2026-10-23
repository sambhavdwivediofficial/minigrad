# MiniGrad Examples

Every script in `examples/`, what it demonstrates, the exact command to run it, and — where applicable — the actual output it produces. For what each API call does, see [`docs/api.md`](api.md). For how the underlying engine works, see [`docs/architecture.md`](architecture.md).

All commands below assume you're in the project root with the package installed:

```powershell
python -m pip install -e ".[dev,viz]"
```

---

## 1. `xor_model.py` + `train_xor.py` + `predict_xor.py` + `evaluate_xor.py`

The original end-to-end walkthrough: a synthetic XOR-style binary classification task (`label = (x1 * x2) > 0`), trained with a hand-rolled training loop (this is the loop that `minigrad.engine.Trainer` later generalized — see [`docs/architecture.md`](architecture.md#6-trainer-the-loop-decoupled-from-the-model)).

**Train:**
```powershell
python .\examples\train_xor.py
```
Trains a `Linear(2,16) -> Tanh -> Linear(16,16) -> Tanh -> Linear(16,1)` network with `Adam`, tracks train/validation loss every epoch, keeps the best validation checkpoint, early-stops on a patience window, and reports held-out test metrics. Saves to `artifacts/xor_model.npz`.

**Predict one input pair:**
```powershell
python .\examples\predict_xor.py 0.8 0.4
```
Loads the checkpoint, runs one forward pass, prints the class-1 probability and predicted class. Same-sign inputs → class 1; opposite-sign → class 0.

**Evaluate on a fresh synthetic set:**
```powershell
python .\examples\evaluate_xor.py
python .\examples\evaluate_xor.py --samples 5000 --seed 42
python .\examples\evaluate_xor.py --threshold 0.6
```
Reports loss, accuracy, precision, recall, specificity, F1, and a full confusion matrix. Typical result: **~99% accuracy**, confirming the autograd engine and training loop are correct end-to-end — this is the project's original regression test, in script form.

---

## 2. `regression_demo.py`

Demonstrates: **axis-aware `Tensor` ops + `huber_loss` + `StepLR` + the reusable `Trainer`/`DataLoader`**, on a genuinely nonlinear task (not just a linear-separable toy).

```powershell
python .\examples\regression_demo.py
python .\examples\regression_demo.py --plot artifacts\regression_loss.png
python .\examples\regression_demo.py --delta 0.5
```

Fits `y = 2*sin(x) + 0.5*x + noise` with a `Linear(1,32) -> Tanh -> Linear(32,32) -> Tanh -> Linear(32,1)` network, `Huber` loss (robust to the occasional noisy outlier), `Adam` + `StepLR` decay, and reports R² via `minigrad.engine.r_squared`. Typical result: **R² ≈ 0.98** on held-out validation data.

With `--plot`, the training/validation loss curve is saved via `minigrad.visualization.plot_loss_curve`:

![Regression training loss curve](images/regression_loss.png)

Train and validation loss track each other closely across the whole run — no widening gap — which is the visual signature of a model that generalized rather than memorized; `EarlyStopping` triggered once validation loss stopped improving, rather than running the full epoch budget.

---

## 3. `moons_spiral_classification.py`

Demonstrates: **`softmax_cross_entropy` + `multiclass_accuracy` + `Dropout` + non-linearly-separable decision boundaries + `plot_decision_boundary`**.

```powershell
python .\examples\moons_spiral_classification.py --plot
python .\examples\moons_spiral_classification.py --task moons --plot
python .\examples\moons_spiral_classification.py --task spiral --spiral-classes 4 --plot
```

### Two-moons (binary)
Two interleaved crescents — not linearly separable, the classic "does your classifier actually learn a curved boundary" sanity check. Trained with `binary_cross_entropy_with_logits` through a `Linear -> Tanh -> Dropout -> Linear` network. Typical result: **~98% validation accuracy**.

![Two-moons decision boundary](images/moons_boundary.png)

The shaded regions are the trained model's predicted class over the whole input plane; the boundary curves to follow the gap between the two crescents instead of drawing a straight line through them — visual proof the network learned a genuinely nonlinear decision surface, not just a linear split.

### Spiral (multiclass)
Three (or `--spiral-classes N`) interleaving spiral arms — a standard, deliberately hard synthetic benchmark for multiclass classifiers. Trained with `softmax_cross_entropy` through a deeper `Linear -> Tanh -> Linear -> Tanh -> Linear` network outputting one logit per class.

![Spiral decision boundary](images/spiral_boundary.png)

Typical result: **100% validation accuracy** — each of the three colored regions traces its corresponding spiral arm all the way to the center, which is strong evidence `log_softmax`/`softmax_cross_entropy`/multiclass `Tensor.__getitem__` fancy-indexing are all composing correctly (see [`docs/architecture.md`](architecture.md#4-new-primitives-are-composed-not-hand-derived-wherever-possible) for why that composition is the real thing being tested here, not just "did accuracy go up").

---

## 4. `tabular_csv_demo.py`

Demonstrates: **`CSVDataset` + feature/target standardization + `BatchNorm1d` + `AdamW`** — the "real tabular data" path, as opposed to data generated directly as NumPy arrays in-memory.

```powershell
python .\examples\tabular_csv_demo.py
python .\examples\tabular_csv_demo.py --size 5000
```

The script first **synthesizes** a small housing-price CSV (`bedrooms, sqft, age, distance_to_center, price`) so the example is self-contained, then exercises the full tabular pipeline a real project would use:

1. `CSVDataset(path, target_column=-1)` loads and splits it into `.x` / `.y`.
2. Features and target are standardized (zero mean, unit variance) using **training-split statistics only** (no validation-set leakage into the standardization).
3. A `Linear(4,32) -> BatchNorm1d(32) -> ReLU -> Linear(32,16) -> ReLU -> Linear(16,1)` network trains with `AdamW` (decoupled weight decay) and `mse_loss`.

Typical result: **R² ≈ 0.98** (on the standardized target) on held-out validation rows.

---

## Shared machinery across all four examples

| Piece | Where it lives | Used by |
|---|---|---|
| `TensorDataset` / `CSVDataset` / `DataLoader` | `minigrad.data` | regression, moons/spiral, tabular |
| `Trainer` / `EarlyStopping` | `minigrad.engine` | regression, moons/spiral, tabular |
| `save_checkpoint` / `load_checkpoint` | `minigrad.checkpoint` | all four |
| `plot_loss_curve` / `plot_decision_boundary` | `minigrad.visualization` | regression (loss curve), moons/spiral (boundary) |

Only `xor_model.py`'s original scripts still hand-roll their own loop — intentionally left as-is so you can diff it against `minigrad/engine.py` and see exactly what got generalized into the reusable `Trainer`.
