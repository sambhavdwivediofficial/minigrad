# MiniGrad: Known Limitations

## Tensor engine
- `@` (matmul) now supports true batched/broadcasted matmul (any number of leading batch dimensions, broadcast against each other the same way NumPy's own `np.matmul` does); the only remaining restriction is that both operands need at least 2 dimensions. No `einsum`, no sparse tensors.
- `float64` only; no mixed precision, no GPU.
- `log()` is valid for `x > 0` only; out-of-domain inputs follow NumPy's own `nan` / `-inf` behaviour rather than raising.
- `abs()` and `max()` use a sub-gradient convention at non-differentiable points: `abs'(0) = 0`, and ties in `max()` split the incoming gradient equally across every tied position. This is a standard, explicit modelling choice, not an accident.
- `__getitem__` supports NumPy-style basic and fancy/integer indexing; the backward pass uses `np.add.at`, so repeated indices correctly accumulate gradient. See "Second-order autodiff" below for the one place this op is more restricted than the rest of the engine.
- `where(condition, a, b)` treats `condition` as a constant boolean array; gradient flows through the two branches, never through the branch choice itself. This mirrors how every autodiff framework implements piecewise functions.

## Second-order autodiff
- `Tensor.backward(create_graph=True)` makes `.grad` itself a differentiable `Tensor` (instead of a plain NumPy array), so a second `.backward()` call can differentiate through the first — this is what `minigrad.grad`, `minigrad.hvp`, `minigrad.hessian`, and `minigrad.newton_step` are built on.
- This is supported for every op **except** `Tensor.__getitem__` and `Tensor.unfold2d`'s own backward rules: differentiating through indexing/slicing (or the sliding-window extraction `Conv2d`/`MaxPool2d`/`AvgPool2d` use) a second time would require a differentiable scatter-add, which isn't implemented yet. Calling `backward(create_graph=True)` through a forward pass that used either op raises a clear `NotImplementedError` naming the restriction, rather than silently returning a wrong gradient.
- Concretely: `Linear`, `LayerNorm`, `BatchNorm1d`, `Dropout`, `mse_loss`/`mae_loss`/`huber_loss`/`binary_cross_entropy_with_logits`, and a single `RNNCell`/`LSTMCell`/`GRUCell` call all fully support `create_graph=True`. `Conv2d`, `MaxPool2d`, `AvgPool2d`, `Embedding`, `softmax_cross_entropy`, and the `RNN`/`LSTM`/`GRU` *sequence* wrappers (which slice a timestep out of the input) do not yet, because each uses indexing internally.
- `hessian()` and `hvp()` currently accept a single 1-D `Tensor` input only (no multi-tensor or arbitrary-shape Hessians yet). `hessian()` computes one `hvp()` call per dimension (`O(n)` double-backward passes); there is no sparse or low-rank Hessian approximation.
- There is no third-order (or higher) autodiff: the "smooth" ops (`tanh`, `sigmoid`, `exp`, `log`, `**`) recompute their local derivative as a differentiable expression of the input and so do, in principle, support it, but this has not been tested or documented beyond second order.

## Modules
- `BatchNorm1d` assumes 2-D input `[batch, features]` only.
- `Dropout` uses its own NumPy RNG instance per layer (seeded at construction) rather than a single global RNG; this keeps layers independently reproducible but means two `Dropout` layers with the same seed draw identical masks.
- Parameters and buffers are both included in checkpoints; only parameters are returned by `.parameters()`.

## Optimizers / schedulers
- `SGD`, `Adam`, `RMSProp`, and `AdamW` all implement `state_dict()` / `load_state_dict()` (momentum/moment buffers, step count, and hyperparameters), and `minigrad.save_training_state` / `load_training_state` persist model **and** optimizer state together so training can resume with bit-for-bit identical behaviour to an uninterrupted run (see `tests/test_optim.py::test_resumed_training_matches_uninterrupted_training_exactly` and `examples/resume_training_demo.py`). This is separate from `save_checkpoint`/`load_checkpoint`, which still save model state only.
- Schedulers (`StepLR`, `CosineAnnealingLR`, `LinearWarmupLR`) have no `state_dict()` of their own yet; resuming training with a scheduler requires re-deriving its position from the restored epoch count.
- Schedulers step once per call; composing multiple schedulers on one optimizer is the caller's responsibility.

## Data / training engine
- `CSVDataset` accepts clean, fully numeric CSV files only: no missing-value imputation, no categorical encoding, no header-based column selection (indices only).
- `DataLoader` shuffling uses a single `numpy.random.Generator`; it is reproducible given a seed but is not multi-process-safe (there is no multi-process loading at all).
- `Trainer` is single-process, single-device, and holds the entire batch's forward graph in memory; there is no gradient accumulation across batches.

## CNN / RNN / Attention
- `Conv2d` / `MaxPool2d` / `AvgPool2d` now extract their sliding windows with `Tensor.unfold2d`, which uses `numpy.lib.stride_tricks.as_strided` to build every window as a single, zero-copy, vectorized NumPy view (no Python loop over output positions at all). The only remaining Python loop is inside `unfold2d`'s *backward* pass, over `kernel_size * kernel_size` offsets — independent of image resolution, so it stays small even as images grow. `tests/test_conv.py::test_conv2d_matches_reference_loop_implementation` checks the new path is numerically identical to the original loop-based one it replaced. `unfold2d` does not support `create_graph=True` for the same reason `__getitem__` doesn't (see "Second-order autodiff" above) — scattering a `Tensor`'s values back into overlapping input positions has no composed equivalent yet.
- `MultiHeadSelfAttention` uses a single batched `@` across `(batch, heads)` with no Python loop — this was the main thing true batched matmul was added to fix.
- There is still no vectorized/strided fast path for truly large images (beyond what `unfold2d` already buys): no im2col caching across layers, no operator fusion, no GPU. `unfold2d`'s window array is `kernel_size^2` times larger than the input in memory (every overlapping position is materialized), which is the standard im2col memory/speed trade-off — fine for the modestly sized images this project trains on, not a production-grade memory budget for very large images or very large batches.
- `RNN` / `LSTM` / `GRU` unroll their timestep loop in Python (this is inherent to recurrence, not something batched matmul changes); both memory and wall-clock cost grow linearly with sequence length, and there is no truncated-BPTT, no packed/variable-length-sequence support, and no cuDNN-style fused gate computation.
- `Embedding` has no `padding_idx` (a reserved "ignore this token" index) and no pretrained-weight loading helper.
- `TransformerLM` is encoder-only-with-a-causal-mask (a decoder-only / GPT-style stack) and has no key/value caching for incremental generation — `generate()` in both `examples/char_lstm_demo.py` and `examples/char_transformer_demo.py` re-runs (or re-truncates-and-reruns) the sequence each step rather than caching past activations. It applies no dropout inside attention or the feed-forward block.
- `examples/char_lstm_demo.py` and `examples/char_transformer_demo.py` both train with mini-batches via `DataLoader` rather than one single full-corpus forward pass per epoch; an earlier full-batch version of both was numerically correct but drove the Python process's resident memory into the gigabytes on a longer run, because NumPy's allocator does not return freed memory to the OS between many differently-shaped allocations per step — mini-batching keeps each step's peak allocation small and resident memory stable. This is a real, documented characteristic of this engine's define-by-run design (every op allocates fresh arrays; there is no in-place/memory-reuse optimization), not specific to these two examples.
- None of `Conv2d`, the recurrent cells, or attention layers have been benchmarked against PyTorch/JAX equivalents; no claims about relative speed are made or implied.

## Diagnostics (`explain_nan`, `explain_gradient`)
- `explain_nan` reports the *first* node (in forward/topological order) whose data contains a NaN or Inf, and one representative path from the tensor you called it on back to that node. If several independent branches produce NaNs at the same "depth" in the graph, only one is reported — this is a debugging aid for finding *a* root cause quickly, not an exhaustive report of every NaN source.
- Its per-op "likely cause" hints are a small, hand-written lookup table (`log`, `/`, `exp`, `pow(0.5)`, `sigmoid`); an op with no entry still gets correctly located, just without a suggested fix.
- `explain_gradient` requires a scalar output and a single `Tensor` input with `requires_grad=True`; it does not (yet) handle multiple inputs in one call, or a function whose path to the output branches and rejoins in ways more complex than straightforward composition (it reports each op on the path once, not once per path through a diamond).
- Its derivative-rule text is a fixed lookup table keyed by the same `_op` label used for debugging/printing; a future custom op with no matching entry would still be gradient-checked correctly, just described as "(no human-readable rule registered for this op)" instead of with real math.
- Both tools walk the *entire* graph reachable from the tensor passed in, which is cheap for the model sizes in this project but is not optimized for very large graphs (e.g. a full multi-layer Transformer's forward pass) — fine for debugging a suspect subgraph, not meant to run on every training step.

## General
- No model/optimizer serialization format beyond the project's own `.npz` checkpoints.
- No autograd support for complex numbers, sparse tensors, or in-place tensor mutation.
- Error messages favor clarity over exhaustiveness; malformed shapes typically surface as a NumPy broadcasting error rather than a MiniGrad-specific one.

This project is for learning the mechanics of autodiff and small neural networks end to end. It is deliberately not hardened for production use; see `README.md` for the educational scope.
