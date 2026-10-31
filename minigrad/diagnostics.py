from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from .tensor import Tensor
from .testing import gradcheck


def _topo_order(root: Tensor) -> list:
    topo = []
    visited = set()

    def build(node):
        if id(node) in visited:
            return
        visited.add(id(node))
        for parent in node._prev:
            build(parent)
        topo.append(node)

    build(root)
    return topo


_NAN_CAUSE_HINTS = {
    "log": "log(x) is only defined for x > 0; this almost always means a value of 0 or less reached log(). "
           "Common fixes: add a small epsilon before log() (e.g. (x + 1e-8).log()), or use a numerically "
           "stable combined op like .log_softmax() instead of .softmax().log().",
    "/": "division produced inf or nan; the denominator likely reached (or passed through) 0 somewhere. "
         "Check whether a normalization, variance, or count term can legitimately be 0.",
    "pow(0.5)": "a square root (** 0.5) was taken of a negative number. If this came from a variance or "
                "sum-of-squares computation, a tiny negative value from floating-point error is a common "
                "cause; adding a small epsilon before ** 0.5 usually fixes it.",
    "exp": "exp(x) overflowed to inf for a large positive x. If this feeds a sum/softmax afterward, "
           "subtract the max first (.softmax()/.log_softmax() already do this internally) rather than "
           "calling .exp() directly on unbounded logits.",
    "sigmoid": "sigmoid saturated; this alone rarely produces nan/inf, so the nan likely originated "
               "upstream of this op and only became visible here.",
}


@dataclass
class NaNReport:
    found: bool
    op: Optional[str] = None
    shape: Optional[tuple] = None
    bad_count: int = 0
    total_count: int = 0
    parent_ops: list = field(default_factory=list)
    hint: Optional[str] = None
    path_from_root: list = field(default_factory=list)

    def summary(self) -> str:
        if not self.found:
            return "No NaN or Inf found anywhere in this computation graph."
        lines = []
        lines.append(f"NaN/Inf first appears at a '{self.op}' operation, output shape {self.shape}.")
        lines.append(f"  {self.bad_count} of {self.total_count} values are NaN/Inf at this node.")
        if self.parent_ops:
            lines.append(f"  Input to this op came from: {', '.join(self.parent_ops)} (all still finite).")
        else:
            lines.append("  This is a leaf tensor (no parents) — the bad values were already in its own data.")
        if self.hint:
            lines.append(f"  Likely cause: {self.hint}")
        if self.path_from_root:
            lines.append(f"  Path from the tensor you called explain_nan() on: {' -> '.join(self.path_from_root)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def explain_nan(tensor: Tensor) -> NaNReport:
    topo = _topo_order(tensor)

    first_bad = None
    for node in topo:
        bad_mask = ~np.isfinite(node.data)
        if bad_mask.any():
            first_bad = node
            break

    if first_bad is None:
        return NaNReport(found=False)

    bad_mask = ~np.isfinite(first_bad.data)
    parent_ops = [p._op or "leaf" for p in first_bad._prev]
    hint = _NAN_CAUSE_HINTS.get(first_bad._op)

    path = []
    node = first_bad
    seen = {id(first_bad)}
    while True:
        label = node._op or "input"
        path.append(label)
        if id(node) == id(tensor):
            break
        child = None
        for candidate in topo:
            if any(id(p) == id(node) for p in candidate._prev) and id(candidate) not in seen:
                child = candidate
                break
        if child is None:
            break
        seen.add(id(child))
        node = child

    return NaNReport(
        found=True,
        op=first_bad._op or "input",
        shape=first_bad.data.shape,
        bad_count=int(bad_mask.sum()),
        total_count=int(bad_mask.size),
        parent_ops=parent_ops,
        hint=hint,
        path_from_root=list(reversed(path)),
    )


_OP_DERIVATIVE_RULES = {
    "+": "sum rule: d/dx(a + b) = 1 for each addend",
    "neg": "d/dx(-x) = -1",
    "*": "product rule: d/da(a * b) = b, d/db(a * b) = a",
    "/": "quotient rule: d/da(a / b) = 1/b,  d/db(a / b) = -a / b^2",
    "matmul": "d/dA(A @ B) = dOut @ B^T,  d/dB(A @ B) = A^T @ dOut",
    "sum": "d/dx(sum(x)) = 1 for every element (broadcasts back to x's shape)",
    "max": "derivative is 1 at the position(s) that achieved the maximum, 0 elsewhere (ties split evenly)",
    "reshape": "derivative passes straight through, reshaped back to the original shape",
    "transpose": "derivative passes straight through, transposed back by the inverse permutation",
    "getitem": "derivative scatters back to exactly the selected positions (accumulates on repeated indices)",
    "unfold2d": "derivative scatter-adds each window's contribution back to every input position it overlapped",
    "relu": "d/dx relu(x) = 1 where x > 0, else 0",
    "tanh": "d/dx tanh(x) = 1 - tanh(x)^2",
    "sigmoid": "d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))",
    "softplus": "d/dx softplus(x) = sigmoid(x)",
    "exp": "d/dx exp(x) = exp(x)",
    "log": "d/dx log(x) = 1 / x",
    "abs": "d/dx |x| = sign(x), defined as 0 at x = 0",
    "where": "derivative routes entirely to whichever branch (a or b) was selected at each position",
    "concat": "derivative is sliced back out to exactly the region each input contributed",
}


def _rule_for(op: str) -> str:
    if op in _OP_DERIVATIVE_RULES:
        return _OP_DERIVATIVE_RULES[op]
    if op.startswith("pow("):
        power = op[len("pow(") : -1]
        return f"power rule: d/dx(x^{power}) = {power} * x^({power}-1)"
    return "(no human-readable rule registered for this op)"


@dataclass
class GradientDerivation:
    steps: list
    verified: bool
    max_absolute_error: float
    max_relative_error: float

    def summary(self) -> str:
        lines = [f"Gradient derivation, {len(self.steps)} op(s) on the path from input to output:", ""]
        for i, (op, rule) in enumerate(self.steps, start=1):
            lines.append(f"  Step {i}: {op:<10s} {rule}")
        lines.append("")
        lines.append(
            "These combine multiplicatively, back to front (the chain rule) — exactly what "
            "Tensor.backward() computes."
        )
        lines.append("")
        if self.verified:
            lines.append(
                f"Verification: gradcheck PASSED (max absolute error {self.max_absolute_error:.2e}, "
                f"max relative error {self.max_relative_error:.2e})."
            )
            lines.append(
                "This confirms the chain above is not just plausible — it matches a finite-difference "
                "approximation of the true derivative to numerical precision."
            )
        else:
            lines.append(
                f"Verification: gradcheck FAILED (max absolute error {self.max_absolute_error:.2e}, "
                f"max relative error {self.max_relative_error:.2e}) — do not trust this derivation."
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def explain_gradient(fn, x: Tensor, eps: float = 1e-6, atol: float = 1e-5, rtol: float = 1e-3) -> GradientDerivation:
    if not x.requires_grad:
        raise ValueError("explain_gradient requires x.requires_grad=True")

    output = fn(x)
    if not isinstance(output, Tensor) or output.data.size != 1:
        raise ValueError("fn(x) must return a scalar Tensor")

    topo = _topo_order(output)
    ancestor_ids = set()
    for node in topo:
        if id(node) == id(x) or any(id(p) in ancestor_ids for p in node._prev):
            ancestor_ids.add(id(node))

    steps = []
    for node in topo:
        if id(node) in ancestor_ids and id(node) != id(x) and node._op:
            steps.append((node._op, _rule_for(node._op)))

    result = gradcheck(fn, x, eps=eps, atol=atol, rtol=rtol)

    return GradientDerivation(
        steps=steps,
        verified=result.passed,
        max_absolute_error=result.max_absolute_error,
        max_relative_error=result.max_relative_error,
    )
