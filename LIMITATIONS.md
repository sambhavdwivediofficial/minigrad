# MiniGrad: Known Limitations

## Tensor engine
- `@` (matmul) supports 2-D operands only. No batched matmul yet.
- `float64` only; no mixed precision, no GPU.
- `log()` is valid for `x > 0` only; out-of-domain inputs follow NumPy's own `nan` / `-inf` behaviour rather than raising.
- `abs()` and `max()` use a sub-gradient convention at non-differentiable points: `abs'(0) = 0`, and ties in `max()` split the incoming gradient equally across every tied position. This is a standard, explicit modelling choice, not an accident.
- `__getitem__` supports NumPy-style basic and fancy/integer indexing; the backward pass uses `np.add.at`, so repeated indices correctly accumulate gradient.
- `where(condition, a, b)` treats `condition` as a constant boolean array; gradient flows through the two branches, never through the branch choice itself. This mirrors how every autodiff framework implements piecewise functions.
- No batched/broadcasted matmul, no `einsum`, no sparse tensors.

## Modules
- `BatchNorm1d` assumes 2-D input `[batch, features]` only.
- `Dropout` uses its own NumPy RNG instance per layer (seeded at construction) rather than a single global RNG; this keeps layers independently reproducible but means two `Dropout` layers with the same seed draw identical masks.
- Parameters and buffers are both included in checkpoints; only parameters are returned by `.parameters()`.

## Optimizers / schedulers
- No optimizer-state checkpointing (only model parameters/buffers are saved); resuming training restarts optimizer moment estimates from zero.
- Schedulers (`StepLR`, `CosineAnnealingLR`, `LinearWarmupLR`) step once per call; composing multiple schedulers on one optimizer is the caller's responsibility.

## Data / training engine
- `CSVDataset` accepts clean, fully numeric CSV files only: no missing-value imputation, no categorical encoding, no header-based column selection (indices only).
- `DataLoader` shuffling uses a single `numpy.random.Generator`; it is reproducible given a seed but is not multi-process-safe (there is no multi-process loading at all).
- `Trainer` is single-process, single-device, and holds the entire batch's forward graph in memory; there is no gradient accumulation across batches.

## CNN / RNN / Attention
- `Conv2d` / `MaxPool2d` / `AvgPool2d` build their output with a Python `for` loop over output spatial positions (im2col-by-slicing, see [`docs/architecture.md`](docs/architecture.md)) rather than a vectorized/strided implementation. This is correct and fully gradient-tested, but materially slower than a real framework's fused kernel — fine for the small images in `examples/cnn_shapes_classification.py`, not for anything approaching real photo resolution.
- There is still no batched matmul (`@` remains 2-D only, as noted above). `MultiHeadSelfAttention` works around this by looping over `(batch, head)` pairs in Python rather than lifting the restriction — same trade-off as the convolution loops, made for the same reason (composition over new hand-derived, harder-to-verify code).
- `RNN` / `LSTM` / `GRU` unroll their timestep loop in Python; both memory and wall-clock cost grow linearly with sequence length, and there is no truncated-BPTT, no packed/variable-length-sequence support, and no cuDNN-style fused gate computation.
- `Embedding` has no `padding_idx` (a reserved "ignore this token" index) and no pretrained-weight loading helper.
- `TransformerLM` is encoder-only-with-a-causal-mask (a decoder-only / GPT-style stack), has no key/value caching for incremental generation (`generate()` in `examples/char_lstm_demo.py`'s LSTM example re-runs the full sequence each step; an equivalent Transformer generator would too), and applies no dropout inside attention or the feed-forward block.
- None of `Conv2d`, the recurrent cells, or attention layers have been benchmarked against PyTorch/JAX equivalents; no claims about relative speed are made or implied.

## General
- No model/optimizer serialization format beyond the project's own `.npz` checkpoints.
- No autograd support for complex numbers, sparse tensors, or in-place tensor mutation.
- Error messages favor clarity over exhaustiveness; malformed shapes typically surface as a NumPy broadcasting error rather than a MiniGrad-specific one.

This project is for learning the mechanics of autodiff and small neural networks end to end. It is deliberately not hardened for production use; see `README.md` for the educational scope and the roadmap for where it is headed next.
