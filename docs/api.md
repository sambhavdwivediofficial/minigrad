# MiniGrad API Reference

Every public symbol in `minigrad`, grouped by module. For the reasoning behind the design, see [`docs/architecture.md`](architecture.md). For runnable code, see [`docs/examples.md`](examples.md). All symbols below are importable directly from `minigrad` (e.g. `from minigrad import Tensor, Linear, Trainer`), except where noted.

Each entry marks whether its backward rule is **hand-derived** (a primitive gradient rule, individually gradient-checked) or **composed** (built from other already-tested primitives — see [`docs/architecture.md`](architecture.md#4-new-primitives-are-composed-not-hand-derived-wherever-possible)).

---

## `minigrad.tensor`

### `class Tensor(data, requires_grad=False, _children=(), _op="")`
The core autodiff primitive: a NumPy array plus a gradient buffer and a position in the computation graph.

**Constructor**
| Parameter | Type | Meaning |
|---|---|---|
| `data` | array-like | Converted to `np.float64` via `np.asarray`. |
| `requires_grad` | `bool` | If `True`, `.grad` is allocated (zeros) and this tensor accumulates gradient during `.backward()`. |
| `_children`, `_op` | internal | Set automatically by operators; not meant to be passed by user code. |

**Properties**
- `.shape` → `tuple` — `data.shape`.
- `.ndim` → `int` — `data.ndim`.
- `.T` → `Tensor` — shorthand for `.transpose()` (full axis reversal). *Composed.*

**Methods**
- `.item() -> float` — scalar value; requires `data.size == 1`.
- `.numpy() -> np.ndarray` — a **copy** of `.data` (safe to mutate without affecting the graph).
- `.detach() -> Tensor` — a new leaf `Tensor` with `requires_grad=False`, copied data, no graph history. Use this before feeding a result into metrics/plots so you don't accidentally build graph you'll never call `.backward()` on.
- `.zero_grad()` — resets `.grad` to zeros (no-op if `requires_grad=False`).
- `.backward(gradient=None)` — runs reverse-mode autodiff from this tensor back through the graph. `gradient=None` is only valid for scalar tensors (defaults to `1.0`); non-scalar tensors require an explicit `gradient` array of matching shape (vector-Jacobian product convention). Raises `RuntimeError` if `requires_grad=False`, raises `ValueError` on a shape mismatch. See [`docs/architecture.md`](architecture.md#2-backward-topological-sort-then-walk-it-in-reverse).

**Arithmetic operators** *(all hand-derived; all support a plain Python/NumPy scalar or array on the other side, auto-wrapped into a `Tensor`)*
`+`, `-` (binary and unary), `*`, `/`, `** power` (scalar exponent only — `TypeError` on a non-scalar power), `@` (matmul, **2-D operands only** — `ValueError` otherwise).

**Reductions**
- `.sum(axis=None, keepdims=False) -> Tensor` — *hand-derived.* Axis-aware; `axis=None, keepdims=False` (the default) matches the original no-argument behaviour exactly.
- `.mean(axis=None, keepdims=False) -> Tensor` — *composed* (`sum / count`). Raises `ValueError` on an empty tensor.
- `.max(axis=None, keepdims=False) -> Tensor` — *hand-derived.* Sub-gradient convention: ties split the incoming gradient equally across every tied position.
- `.argmax(axis=None)` — **not differentiable**, returns a plain `np.ndarray` of indices, not a `Tensor`.

**Shape ops**
- `.reshape(*shape) -> Tensor` — *hand-derived* (backward is just reshaping `out.grad` back to the original shape).
- `.transpose(*axes) -> Tensor` — *hand-derived.* No arguments reverses all axes (like `np.transpose`).
- `__getitem__(index) -> Tensor` — *hand-derived.* Supports NumPy basic and fancy/integer indexing. Backward uses `np.add.at`, so **repeated indices correctly accumulate** gradient (this is what makes an embedding-style gather differentiable).

**Nonlinearities** *(all hand-derived except where noted)*
- `.relu()`, `.tanh()`, `.sigmoid()` (numerically stable, branch on sign), `.softplus()` (via `np.logaddexp`), `.exp()`, `.log()` (domain `x > 0`; see [`LIMITATIONS.md`](../LIMITATIONS.md)), `.abs()` (sub-gradient `abs'(0) = 0`).
- `.log_softmax(axis=-1) -> Tensor` — *composed*, numerically stable (`x - max(x) - log(sum(exp(x - max(x))))`).
- `.softmax(axis=-1) -> Tensor` — *composed* (`log_softmax(...).exp()`).

### Module-level graph functions
- `where(condition: np.ndarray, a: Tensor, b: Tensor) -> Tensor` — *hand-derived.* `condition` is a **plain boolean NumPy array**, not a Tensor — the branch *choice* is a constant, gradient flows through the two branches only.
- `concatenate(tensors: Sequence[Tensor], axis=0) -> Tensor` — *hand-derived* (backward slices `out.grad` back into each input's region).
- `stack(tensors: Sequence[Tensor], axis=0) -> Tensor` — *composed*: `reshape` each input to insert a new size-1 axis, then `concatenate`.

---

## `minigrad.nn`

### `class Module`
Base class for every layer. Not instantiated directly.

- `.training: bool` — defaults to `True`.
- `.train(mode=True) -> Module` / `.eval() -> Module` — recursively sets `.training` on `self` and every nested `Module` (through attributes, lists, tuples, and dicts). Returns `self` for chaining.
- `.register_buffer(name, value)` / `.buffer(name) -> np.ndarray` / `.set_buffer(name, value)` — persistent non-trainable state (see [`docs/architecture.md`](architecture.md#5-module-parameters-vs-buffers)).
- `.named_parameters(prefix="")` / `.parameters()` — recursively yields every trainable `Tensor`, dotted-name style (e.g. `"layers.0.weight"`).
- `.named_buffers(prefix="")` — recursively yields every registered buffer the same way.
- `.zero_grad()` — calls `.zero_grad()` on every parameter.
- `.num_parameters() -> int` — total trainable element count.
- `.state_dict() -> dict` — parameters **and** buffers (buffer keys prefixed `__buffer__.` internally).
- `.load_state_dict(state, strict=True)` — restores both; `strict=True` raises `ValueError` on any missing/extra parameter key (buffers are always loaded best-effort).

### `class Linear(in_features, out_features, seed=None, init="he")`
`forward(x) -> x @ weight + bias`. `init` is `"he"` (default, `sqrt(2/in_features)`), `"xavier"` (`sqrt(1/in_features)`), or `"zeros"`.

### `class ReLU`, `class Tanh`, `class Sigmoid`
Stateless wrappers around `Tensor.relu()` / `.tanh()` / `.sigmoid()`.

### `class Flatten`
`forward(x)` collapses every dimension after the batch dimension: `[batch, d1, d2, ...] -> [batch, d1*d2*...]`.

### `class Dropout(p=0.5, seed=None)`
Inverted dropout. At `training=True` and `p > 0`: zeroes each element with probability `p`, rescales survivors by `1/(1-p)`. At `training=False`, or `p == 0`, it's the identity. Raises `ValueError` if `p` is not in `[0, 1)`. Each instance owns its own `np.random.Generator`.

### `class LayerNorm(normalized_shape, eps=1e-5)`
Normalizes each sample over its **last** dimension, then applies a learned `gamma`/`beta` (both shape `[1, normalized_shape]`). Identical behaviour in train and eval mode.

### `class BatchNorm1d(num_features, eps=1e-5, momentum=0.1)`
Normalizes each feature over the **batch** dimension at train time (and updates the `running_mean` / `running_var` buffers with the given `momentum`); at eval time, normalizes using the stored running statistics instead. 2-D input `[batch, num_features]` only.

### `class Sequential(*layers)`
`forward(x)` applies every layer in order. `.layers` is a plain Python list (so `model.layers[i]` works).

### Loss functions
| Function | Signature | Notes |
|---|---|---|
| `mse_loss` | `(prediction, target) -> Tensor` | `((prediction - target) ** 2).mean()`. *Composed.* |
| `mae_loss` | `(prediction, target) -> Tensor` | `(prediction - target).abs().mean()`. *Composed.* |
| `huber_loss` | `(prediction, target, delta=1.0) -> Tensor` | Quadratic for `|residual| <= delta`, linear beyond. *Composed* via `where`. |
| `binary_cross_entropy_with_logits` | `(logits, target) -> Tensor` | Numerically stable (`softplus(x) - x*target`). `target` is 0/1-valued. *Composed.* |
| `softmax_cross_entropy` | `(logits, targets) -> Tensor` | `logits: [batch, num_classes]`, `targets`: **plain integer array**, shape `[batch]` (not one-hot, not a `Tensor`). Raises `ValueError` on a bad `targets` shape. *Composed.* |
| `kl_divergence` | `(log_probs, target_probs) -> Tensor` | `KL(target || predicted)`. `log_probs` is a differentiable `Tensor` (e.g. from `.log_softmax()`); `target_probs` is a fixed (non-trainable) distribution. *Composed.* |

---

## `minigrad.optim`

### `class Optimizer(parameters)`
Base class. Raises `ValueError` if `parameters` is empty. `.zero_grad()` resets every parameter's gradient.

### `class SGD(parameters, lr=1e-2, momentum=0.0)`
Classic (optionally momentum) SGD. Validates `lr > 0` and `0 <= momentum < 1`.

### `class Adam(parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)`
Standard Adam with bias-corrected moment estimates.

### `class RMSProp(parameters, lr=1e-2, alpha=0.99, eps=1e-8, momentum=0.0)`
Exponential moving average of squared gradients; optional momentum on top of the scaled update.

### `class AdamW(parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)`
Adam with **decoupled** weight decay (Loshchilov & Hutter): decay is applied directly to the parameter (`p -= lr * weight_decay * p`), not mixed into the gradient the way "Adam + L2 penalty" does.

### `clip_grad_norm_(parameters, max_norm) -> float`
In-place gradient clipping by global L2 norm across all given parameters. Returns the **pre-clipping** total norm (useful for logging).

### Schedulers — `StepLR(optimizer, step_size, gamma=0.1)`, `CosineAnnealingLR(optimizer, t_max, eta_min=0.0)`, `LinearWarmupLR(optimizer, warmup_steps)`
All three mutate `optimizer.lr` in place on each `.step()` call (call once per epoch or per batch, matching how you schedule). `.get_last_lr()` reads the current value back.

---

## `minigrad.data`

### `class Dataset`
Minimal map-style interface: subclasses implement `__len__` and `__getitem__`.

### `class TensorDataset(*arrays)`
Wraps one or more same-length NumPy arrays, e.g. `TensorDataset(features, labels)`. Raises `ValueError` if the arrays' first dimensions disagree.

### `class CSVDataset(path, target_column=-1, feature_columns=None, has_header=True, delimiter=",")`
Loads a **fully numeric** CSV into `(features, target)` pairs via `np.genfromtxt`. Raises `ValueError` on any missing/non-numeric value (see [`LIMITATIONS.md`](../LIMITATIONS.md)). `.x` / `.y` hold the full loaded arrays.

### `class DataLoader(dataset, batch_size=32, shuffle=False, seed=None, drop_last=False)`
Yields `(x_batch, y_batch, ...)` tuples of plain NumPy arrays. Shuffling uses a seeded `np.random.Generator` (reproducible). Has a vectorized fast path when `dataset` is a `TensorDataset` (fancy-indexes the whole batch at once instead of looping per-sample). `len(loader)` returns the number of batches per epoch.

---

## `minigrad.engine`

### `class History`
Plain dataclass: `.train_loss`, `.val_loss`, `.train_metrics`, `.val_metrics` — each a list, one entry appended per epoch.

### `class EarlyStopping(patience=10, mode="min", min_delta=0.0)`
`.step(value) -> bool` (returns whether `value` was a new best) and sets `.should_stop = True` once `patience` consecutive non-improving calls have occurred. `mode="max"` for metrics where higher is better (e.g. accuracy).

### Metric functions (for `Trainer(metric_fns=...)`)
- `binary_accuracy(predictions, targets, threshold=0.0) -> float` — for `binary_cross_entropy_with_logits`-style models (`predictions` are raw logits).
- `multiclass_accuracy(predictions, targets) -> float` — for `softmax_cross_entropy`-style models (`predictions: [batch, num_classes]` logits, `targets`: integer class indices).
- `r_squared(predictions, targets) -> float` — coefficient of determination, for regression.

### `class Trainer(model, optimizer, loss_fn, metric_fns=None, scheduler=None, grad_clip=None)`
`loss_fn(predictions, targets) -> Tensor`; `metric_fns`: `dict[str, Callable[[Tensor, targets], float]]`.

`.fit(train_loader, val_loader=None, epochs=100, early_stopping=None, checkpoint_path=None, verbose_every=1) -> History` — runs the full train/validate loop, restores the best (lowest `val_loss`) `state_dict()` at the end if a `val_loader` was given, and saves a checkpoint if `checkpoint_path` is given. See [`docs/architecture.md`](architecture.md#6-trainer-the-loop-decoupled-from-the-model).

---

## `minigrad.checkpoint`

- `save_checkpoint(model: Module, path) -> None` — `np.savez_compressed` of `model.state_dict()` (parameters **and** buffers).
- `load_checkpoint(model: Module, path) -> None` — loads and calls `model.load_state_dict(...)`.

---

## `minigrad.testing`

### `class GradcheckResult`
Frozen dataclass: `.passed: bool`, `.max_absolute_error: float`, `.max_relative_error: float`.

### `gradcheck(fn, x: Tensor, eps=1e-6, atol=1e-5, rtol=1e-3) -> GradcheckResult`
Central finite-difference check of a single-`Tensor`-input, scalar-`Tensor`-output function against MiniGrad's own `.backward()`. `x` must have `requires_grad=True`.

### `gradcheck_multi(fn, tensors, eps=1e-6, atol=1e-5, rtol=1e-3) -> list[GradcheckResult]`
Same idea for a function of **several** `Tensor` inputs (e.g. `matmul`, `concatenate`, `where`) — one `GradcheckResult` per input tensor, in the order given.

---

## `minigrad.visualization`
*(Not imported by `minigrad/__init__.py` — import explicitly: `from minigrad.visualization import ...`. Requires the optional `viz` extra: `pip install -e ".[viz]"`.)*

- `plot_loss_curve(history: History, path, title="Training loss") -> None` — saves a PNG of `history.train_loss` (and `history.val_loss`, if present).
- `plot_decision_boundary(model, x, y, path, resolution=200, title="Decision boundary", multiclass=False) -> None` — for a `[N, 2] -> logits` model. `multiclass=False` (default) uses `sign(logit)`; `multiclass=True` uses `argmax` over multiple output logits.

Both raise a clear `ImportError` (not a matplotlib traceback) if matplotlib isn't installed.
