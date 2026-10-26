# MiniGrad Architecture

This document explains *how* MiniGrad works internally: how a `Tensor` builds a computation graph, how `backward()` turns that graph into gradients, and how the layers on top (`nn`, `optim`, `data`, `engine`) are assembled from that one primitive. For *what* each function does, see [`docs/api.md`](api.md). For runnable demonstrations, see [`docs/examples.md`](examples.md).

## 1. The core idea: a tape of local derivatives

Every numerical framework with automatic differentiation is built on one idea: **if you know the local derivative of every small operation, the chain rule lets you compose them into the derivative of an arbitrarily large expression.** MiniGrad implements this the simplest possible way — define-by-run, reverse mode, no compilation step.

A `Tensor` is a NumPy array plus four extra fields:

```text
Tensor
 ├── data            the actual numbers (np.ndarray, float64)
 ├── requires_grad   should this tensor accumulate a gradient?
 ├── grad            the accumulated gradient (same shape as data, or None)
 ├── _prev            the tensors that produced this one (its parents)
 ├── _op              a short label, for debugging/printing
 └── _backward        a closure: "given out.grad, add my share into each parent's .grad"
```

`_prev` and `_backward` are the whole trick. Every operator (`+`, `*`, `matmul`, `relu`, `reshape`, ...) does two things when called:

1. Computes the forward result eagerly, in NumPy, right away (this is "define-by-run" — there is no separate graph-building phase).
2. Wires up a `Tensor` for that result whose `_prev` points at the input tensor(s), and whose `_backward` closure captures everything it needs (the inputs, the output's own data, broadcasting shapes) to compute *local* derivatives once called.

```python
def __mul__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data * other.data, self.requires_grad or other.requires_grad, (self, other), "*")

    def _backward():
        if self.requires_grad:
            self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
        if other.requires_grad:
            other.grad += _unbroadcast(self.data * out.grad, other.data.shape)

    out._backward = _backward
    return out
```

Notice what `_backward` does *not* do: it does not recurse into `self._backward()` or `other._backward()`. It only computes **this operation's** local contribution and adds it into the parents' `.grad`. Who calls every node's `_backward` exactly once, in the right order, is a separate concern — that's `backward()`.

## 2. `backward()`: topological sort, then walk it in reverse

```python
def backward(self, gradient=None):
    ...
    topo: list[Tensor] = []
    visited: set[int] = set()

    def build(node):
        if id(node) in visited:
            return
        visited.add(id(node))
        for parent in node._prev:
            build(parent)
        topo.append(node)

    build(self)
    self.grad += gradient
    for node in reversed(topo):
        node._backward()
```

`build()` is a depth-first post-order traversal: a node is only appended to `topo` *after* all of its parents have been appended. That guarantees `topo` is a valid topological order of the DAG. Reversing it means: **process the loss first, then everything that fed the loss, and so on backward to the leaves.** By the time any node's `_backward()` runs, every node that depends on it (every "child" in the forward graph) has already deposited its contribution into that node's `.grad` — so `.grad` is complete and correct the moment it is read.

This is why a tensor used in **more than one place** still gets the right gradient: `_prev` can list it as a parent of several different output nodes, `visited` ensures it is only scheduled once in `topo`, but every one of those output nodes' `_backward()` calls independently does `self.grad += ...` — the contributions *accumulate* rather than overwrite. `tests/test_autograd.py::test_shared_tensor_contributes_through_multiple_paths` exists specifically to pin this behaviour down.

## 3. Broadcasting: forward is automatic, backward is not

NumPy broadcasts `(3, 4) + (1, 4)` to `(3, 4)` automatically. Autodiff has to **undo** that broadcast on the way back: a gradient that arrives with shape `(3, 4)` needs to be summed back down to `(1, 4)` before it can be added into a parameter that only had shape `(1, 4)` to begin with (otherwise the shapes would not even match, and semantically, "all the positions that one row fed" must have their gradients summed into that row). That is exactly what `_unbroadcast` does:

```python
def _unbroadcast(grad, target_shape):
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(target_shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(target_shape)
```

Every binary op (`+`, `*`, `/`, `where`) routes its backward contribution through `_unbroadcast`. Axis-reducing ops (`sum`, `mean`, `max`) go the other way: `_expand_reduced_grad` re-inserts the size-1 dimensions a `keepdims=False` reduction removed, so the stored `out.grad` can broadcast back up against the original input shape. `tests/test_autograd.py` sweeps a matrix of shape combinations against both directions.

`@` (matmul) has its own version of this, `_unbroadcast_matmul`, because matmul broadcasting is not quite the same rule: only the *leading batch dimensions* of `np.matmul(a, b)` broadcast (the way `(4, 1, 3, 5) @ (4, 6, 5, 2)` broadcasts to batch shape `(4, 6)`); the trailing two dimensions are the actual matrix multiply and are never broadcast. `_unbroadcast_matmul` is the same while-loop-then-per-axis-sum idea as `_unbroadcast`, just applied to `grad.shape[:-2]` against `target_shape[:-2]` and leaving the last two dimensions alone.

## 4. New primitives are *composed*, not hand-derived, wherever possible

A hand-derived backward rule is the highest-risk code in the whole project — one sign error and training silently diverges. MiniGrad's strategy for every "advanced" primitive added after the original five ops (`+ - * / matmul`) is: **derive it in terms of ops that are already gradient-tested, and let the chain rule do the rest.**

- `log_softmax(x) = (x - max(x)) - log(sum(exp(x - max(x))))` is written as literal Tensor arithmetic (`sub`, `max`, `exp`, `sum`, `log`) — not a single fused kernel with a hand-written Jacobian. Its correctness is a *consequence* of `max`, `exp`, `sum`, and `log` each being individually correct.
- `softmax_cross_entropy` is `-log_softmax(logits)[batch_indices, targets].mean()` — i.e. `log_softmax` plus fancy-indexing (`__getitem__`) plus `mean`. No cross-entropy-specific backward code exists anywhere.
- `LayerNorm.forward` and `BatchNorm1d.forward` are plain Tensor expressions using `mean(axis=...)`, `**0.5`, `/` — not a fused normalization kernel.
- `huber_loss` is `where(|diff| <= delta, quadratic_branch, linear_branch).mean()`, where `where` itself has exactly one hand-derived backward rule (route the gradient to whichever branch was selected, treating the boolean mask as a constant).
- `stack(tensors, axis)` is implemented as "insert a size-1 axis into each tensor via `reshape`, then `concatenate`" — it has no backward rule of its own at all.

This is deliberate: every genuinely new hand-written backward rule (there are only a handful — `+ *  / ** matmul sum max reshape transpose getitem exp log abs where concatenate`) is small enough to gradient-check in isolation, and everything else inherits correctness for free. `docs/api.md` marks which primitives are hand-derived vs. composed.

## 5. `Module`: parameters vs. buffers

A `Module` tree (e.g. `Sequential(Linear(...), BatchNorm1d(...), ...)`) is walked recursively by `named_parameters()` / `named_buffers()`, which descend through `Module.__dict__`, lists, tuples, and dicts looking for:

- **`Tensor` instances with `requires_grad=True`** → these are *parameters*. They are what `optimizer.step()` updates and what `.zero_grad()` resets.
- **plain NumPy arrays registered via `register_buffer()`** → these are *buffers*. `BatchNorm1d`'s running mean/variance are the only example in this codebase. Buffers are *not* touched by any optimizer, but they *are* part of the model's state — `state_dict()` / `load_state_dict()` include both, prefixing buffer keys with `__buffer__.` internally so a checkpoint round-trips both kinds of state without collision.

`Module.train(mode)` / `.eval()` walk the same tree and flip every submodule's `self.training` flag recursively — this is what makes `Dropout` turn into the identity function and `BatchNorm1d` switch from "normalize using this batch's statistics" to "normalize using the stored running statistics" without the caller needing to touch each layer individually.

## 6. `Trainer`: the loop, decoupled from the model

`Trainer._run_epoch()` is the same ~15 lines every one of the example scripts used to hand-roll independently: iterate a `DataLoader`, zero gradients, forward, compute loss, backward, optionally clip gradients, step the optimizer, accumulate a running loss/metric average. The only things a caller supplies are a `Module`, an `Optimizer`, a `loss_fn(predictions, targets) -> Tensor`, and optionally a dict of `metric_fns`. Because `loss_fn` and `metric_fns` are plain callables, `Trainer` has no idea whether it is doing binary classification, multiclass classification, or regression — that knowledge lives entirely in which loss function the caller passed in. This is why `examples/regression_demo.py`, `examples/moons_spiral_classification.py`, and `examples/tabular_csv_demo.py` share one engine instead of three.

`EarlyStopping` and "keep the best `state_dict()` seen so far, restore it at the end" are implemented inside `Trainer.fit()` for the same reason `train_xor.py` originally implemented them inline — except now every example gets them for free.

## 7. CNNs, RNNs and attention: pushing composition as far as it goes

Section 4 made a claim: most "advanced" primitives should be *composed*, not hand-derived. `Conv2d`, `MaxPool2d`, the recurrent cells, and the attention mechanism are the real test of that claim, because they are exactly the layers a reader would expect to need custom backward math.

**`Conv2d` has zero convolution-specific gradient code.** The forward pass works in three composed steps:
1. *im2col by slicing*: for every output position `(i, j)`, slice out the `[N, C_in, kH, kW]` patch the kernel would see there (plain `Tensor.__getitem__`, already gradient-tested) and flatten it to `[N, C_in*kH*kW]` (`reshape`).
2. *stack the patches* into `[N, H_out*W_out, C_in*kH*kW]`.
3. *one batched matmul* against the flattened `[C_in*kH*kW, C_out]` kernel (broadcasting the kernel across the `N` batch dimension — see Section 8), then `transpose`+`reshape` back to `[N, C_out, H_out, W_out]`.

Every one of those steps already had a gradient test before `Conv2d` existed. `tests/test_conv.py::test_conv2d_gradcheck_all_inputs` is really testing whether the *composition* is wired correctly, not whether any new calculus is correct — there isn't any new calculus. `MaxPool2d` / `AvgPool2d` are the same idea with `.max(axis=...)` / `.mean(axis=...)` standing in for the matmul step. `padding` is implemented as `concatenate`-ing zero tensors around the border (`pad2d`) rather than as its own op. The patch-extraction loop over `(i, j)` is still a Python loop (see `LIMITATIONS.md`) — that part is about gathering overlapping windows, which is a different problem from matmul batching and wasn't changed by Section 8's work.

**The recurrent cells (`RNNCell`, `LSTMCell`, `GRUCell`) are even more direct**: every gate is `(x @ W_x + h @ W_h + b)` through a `sigmoid`/`tanh`, and every cell/hidden-state update is `+`/`*` between the results. `RNN`/`LSTM`/`GRU` then just call the cell once per timestep in a Python `for` loop and `stack` the outputs — this *is* backpropagation-through-time, made visible rather than hidden behind a fused kernel: unrolling the loop literally builds one node per timestep in the same computation graph section 1–2 already described, so `.backward()` differentiates through time for free. This loop is inherent to recurrence (each step genuinely depends on the previous one) and is not a batched-matmul question.

**Multi-head self-attention** composes a single batched `@` across `(batch, heads)` at once, `.swapaxes(-1, -2)`, `.softmax(axis=-1)`, `where` (for the causal mask), and `reshape`/`.transpose()` (to split heads out of `d_model` and merge them back). `MultiHeadSelfAttention.forward` projects `x` to Q/K/V, reshapes each to `[batch, seq_len, num_heads, d_k]` and `.transpose(0, 2, 1, 3)`s to `[batch, num_heads, seq_len, d_k]`, then calls `scaled_dot_product_attention` — itself four lines of composed Tensor arithmetic — exactly **once**, letting batched matmul handle every `(batch, head)` pair at once instead of looping over them in Python (see Section 8 for why that loop used to be necessary and isn't anymore). `TransformerEncoderBlock` wraps that in the standard residual-plus-`LayerNorm` sandwich around attention and a position-wise `FeedForward`, and `TransformerLM` stacks several blocks behind an `Embedding` lookup and a fixed (non-trainable) `sinusoidal_positional_encoding` buffer, with `causal=True` so position `t` can never attend to position `> t` — `tests/test_attention.py::test_causal_mask_blocks_future_positions` checks this directly, by perturbing a future token and confirming earlier outputs are unchanged, not just that gradients check out.

The practical payoff of building things this way: `tests/test_recurrent.py::test_lstm_can_memorize_a_short_repeating_sequence` and `tests/test_attention.py::test_transformer_lm_can_memorize_a_short_repeating_sequence` don't just gradient-check individual pieces — they train `LSTM` and `TransformerLM` on a toy next-character task end to end and assert the loss actually drops near zero. That only works if every composed piece, all the way down to `Tensor.__add__`, is correct — which is exactly the point of insisting on composition over hand-derived shortcuts.

## 8. True batched matmul

The original `@` only accepted two 2-D operands, which is why `Conv2d` had to flatten `[N, H_out*W_out, ...]` down to one giant 2-D matrix and `MultiHeadSelfAttention` had to loop over `(batch, head)` pairs in Python. `@` now mirrors `np.matmul`'s own rule: both operands need at least 2 dimensions, the **last two** dimensions are the actual matrix multiply (`(..., m, n) @ (..., n, p) -> (..., m, p)`), and every dimension *before* that broadcasts the same way `+` or `*` would.

The backward rule is the direct generalization of the 2-D one — `grad_a = out.grad @ b.swapaxes(-1, -2)`, `grad_b = a.swapaxes(-1, -2) @ out.grad` — using `np.matmul` (for plain gradients) so it also broadcasts correctly, then `_unbroadcast_matmul` (Section 3) to undo any batch-dimension broadcasting. This is still one hand-derived rule, not twelve: the same `__matmul__` now just handles 2-D as the special case where there are zero batch dimensions, and `tests/test_multi_input_ops.py` gradient-checks it across 3-D, 4-D, and mixed-rank broadcasting shapes, plus confirms the plain 2-D case is byte-for-byte unaffected.

The payoff, beyond `MultiHeadSelfAttention` losing its loop: `Linear.forward` (`x @ self.weight + self.bias`) now accepts `x` of **any** rank — `[batch, in_features]` or `[batch, seq_len, in_features]` — with zero code changes, because NumPy already treats a plain 2-D weight matrix as implicitly batch-broadcasting against a higher-rank input. `FeedForward` and `TransformerLM.output_projection` used to manually flatten `[batch, seq, d_model]` to 2-D and reshape back around every `Linear` call; that flattening is gone now, since `Linear` just works directly on the 3-D tensor.

## 9. Second-order autodiff: differentiating through `backward()` itself

Every example above produces a first derivative: call `.backward()`, read `.grad`. Section 2 described `.grad` as always ending up a plain NumPy array — which is exactly what makes a *second* derivative impossible to get by calling `.backward()` again: there's no graph left to differentiate, `.grad` is just numbers.

`Tensor.backward(gradient=None, create_graph=False)` adds one new branch to that story. When `create_graph=True`, the seed gradient starts life as a `Tensor` instead of a NumPy array, and `_accumulate_grad` — the single helper every `_backward` closure calls to deposit its contribution — notices the difference: if the incoming contribution is a `Tensor`, it accumulates via `self.grad + contribution` (building graph) instead of `self.grad += contribution` (mutating an array). Every hand-derived closure in `tensor.py` has a `isinstance(out.grad, Tensor)` branch that computes the *same* local derivative, but as Tensor arithmetic (`other * out.grad` instead of `other.data * out.grad`) rather than raw NumPy. Because that Tensor arithmetic goes through the exact same `+`, `*`, `matmul`, `reshape` and so on that are already gradient-tested, the result — `.grad` as a `Tensor` with its own `_prev`/`_backward` — is itself a fully valid node in a new graph, ready for a second `.backward()` call.

Where this is honest about its limits: the smooth ops (`tanh`, `sigmoid`, `exp`, `log`, `**`) recompute their *local derivative* as a differentiable expression of the input itself — e.g. `tanh`'s Tensor-mode backward is `(1 - self.tanh()**2) * out.grad`, using `self.tanh()` (a fresh, differentiable call) rather than the cached forward value — so differentiating through them a second time is mathematically correct, not just plumbing. `max`, `abs`, and `where`'s backward rules multiply by a NumPy-derived constant mask (which positions were the maximum, which branch was selected) exactly as in first-order mode — appropriate, since those masks really are locally constant (the second derivative of a kink is zero almost everywhere) — but it does mean the mask itself was computed once, from the original forward pass, not re-derived. `Tensor.__getitem__` is the one op that does not get a Tensor-mode branch at all: scattering a `Tensor`'s values back into another `Tensor` via `np.add.at`'s pattern has no composed equivalent yet, so its closure raises a clear `NotImplementedError` naming every layer that would be affected (`Conv2d`, `MaxPool2d`/`AvgPool2d`, `Embedding`, `softmax_cross_entropy`, the RNN/LSTM/GRU sequence wrappers) rather than silently returning a wrong gradient. Composed ops built *only* from the smooth/supported primitives — `Linear`, `LayerNorm`, `BatchNorm1d`, `Dropout`, `mse_loss`/`mae_loss`/`huber_loss`/`binary_cross_entropy_with_logits`, a single `RNNCell`/`LSTMCell`/`GRUCell` call — inherit `create_graph=True` support automatically, for the same reason they inherit first-order correctness: they're expressions built from already-covered pieces, not new code paths.

`minigrad/autograd.py` wraps the raw mechanism in a small functional API: `grad(output, inputs, create_graph=False)` calls `.backward()` and reads back `.grad` for the requested inputs; `hvp(fn, x, v)` computes a Hessian-vector product in exactly two backward passes (Pearlmutter's trick — `backward(create_graph=True)` once for the gradient, then one more plain `backward()` on `(grad * v).sum()`) without ever materializing the full Hessian; `hessian(fn, x)` calls `hvp` once per input dimension to build the full matrix; and `newton_step` solves one damped Newton update `x - H⁻¹g` from a single `hessian()` + `grad` call. `tests/test_second_order.py::test_newton_step_reaches_quadratic_minimum_in_one_step` is the sharpest correctness check in the test suite for this feature: Newton's method reaches the *exact* minimum of a quadratic in one step if and only if the Hessian is exactly right, so that test passing is a strong, specific signal — not just "some second derivative came out", but "this second derivative is the real one".

## 10. Where the design intentionally stops

See [`LIMITATIONS.md`](../LIMITATIONS.md) for the full list. The short version: this architecture is optimized for **traceability** (you can read `_backward` and see exactly what it does) over performance (no operator fusion, no kernel compilation, no GPU) or production robustness (no distributed training, no mixed precision, no multi-process data loading). That trade-off is the entire point of an educational framework — see the main [`README.md`](../README.md) for the project's scope.
