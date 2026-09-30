"""Prepared SciPy/C++ propagation with unchanged funded-model event accounting.

The private helpers imported below intentionally share the frozen financial
specification. Agreement tests a numerical implementation, not that specification.
"""

from dataclasses import dataclass
import ctypes
import math
from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded

from qf_research.optimal_stopping.contracts import positive_integer
from qf_research.optimal_stopping.funded_cash import (
    _schedule, _grid, _anchors, _apply, _result, deterministic_value,
)


def _arrays(diffusion, initial, widths, theta, left, right):
    arrays = tuple(np.ascontiguousarray(x, dtype=np.float64) for x in
                   (diffusion, initial, widths, theta, left, right))
    if any(a.ndim != 1 or not np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("Finite one-dimensional arrays required")
    d, v, h, w, lo, hi = arrays
    if len(d) == 0 or len(v) != len(d) + 2 or len(h) == 0:
        raise ValueError("Non-empty interior and step arrays with matching value shape required")
    if any(len(a) != len(h) for a in (w, lo, hi)):
        raise ValueError("Each substep needs a weight and two boundaries")
    if np.any(d < 0) or np.any(h <= 0) or np.any((w < .5) | (w > 1)):
        raise ValueError("Invalid diffusion, positive step width or implicit weight")
    return arrays


def advance_scipy(diffusion, initial, widths, theta, left, right):
    """Readable prepared reference, using SciPy's compiled banded solver."""
    d, v, h, w, lo, hi = _arrays(diffusion, initial, widths, theta, left, right)
    values = v.copy()
    for width, weight, lower, upper in zip(h, w, lo, hi):
        rhs = values[1:-1] + (1-weight)*width*d*(values[:-2]-2*values[1:-1]+values[2:])
        rhs[0] += weight*width*d[0]*lower
        rhs[-1] += weight*width*d[-1]*upper
        band = np.zeros((3, len(d)))
        band[0, 1:] = -weight*width*d[:-1]
        band[1] = 1 + 2*weight*width*d
        band[2, :-1] = -weight*width*d[1:]
        values[1:-1] = solve_banded((1, 1), band, rhs, check_finite=False)
        values[0], values[-1] = lower, upper
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("Non-finite prepared SciPy values")
    return values


class NativeKernel:
    """Load an explicitly supplied local library; no implicit compilation."""

    def __init__(self, library):
        self.library = Path(library).resolve(strict=True)
        self._handle = ctypes.CDLL(str(self.library))
        self._advance = self._handle.qf_funded_advance
        pointer = ctypes.POINTER(ctypes.c_double)
        self._advance.argtypes = [pointer, ctypes.c_size_t, pointer, pointer,
                                  pointer, pointer, pointer, ctypes.c_size_t, pointer]
        self._advance.restype = ctypes.c_int

    def __call__(self, diffusion, initial, widths, theta, left, right):
        d, v, h, w, lo, hi = _arrays(diffusion, initial, widths, theta, left, right)
        output = np.empty_like(v)
        ptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        status = self._advance(ptr(d), len(d), ptr(v), ptr(h), ptr(w), ptr(lo),
                               ptr(hi), len(h), ptr(output))
        if status == 1:
            raise ValueError("Native kernel rejected its input")
        if status == 2:
            raise FloatingPointError("Non-finite native propagation or non-positive pivot")
        if status != 0:
            raise RuntimeError(f"Native propagation failed with status {status}")
        return output


@dataclass(frozen=True)
class Block:
    earlier: float
    widths: np.ndarray
    theta: np.ndarray
    left: np.ndarray
    right: np.ndarray


@dataclass(frozen=True)
class PreparedProblem:
    model: object
    option: object
    rights: tuple
    events: dict
    grid: np.ndarray
    diffusion: np.ndarray
    blocks: tuple[Block, ...]
    time_intervals: int


def prepare(model, option, space_steps, time_steps, y_max, decisions, *, propagation_times=()):
    """Prepare the exact reference grid, damping and boundary schedule once."""
    time_steps = positive_integer(time_steps, "time_steps")
    rights, events = _schedule(model, option, decisions)
    grid = _grid(model, option, rights, events, space_steps, y_max)
    anchors = _anchors(option, rights, events, propagation_times)
    maximum_step = option.maturity/time_steps
    times = [0.]
    for a, b in zip(anchors, anchors[1:]):
        count = max(1, math.ceil((b-a)/maximum_step))
        times.extend(np.linspace(a, b, count+1)[1:])
    diffusion = .5*model.volatility**2*np.arange(1, len(grid)-1, dtype=float)**2

    def boundary(t):
        future = [d for d in rights if d.time > t]
        if option.kind == "put":
            return max(option.strike*d.settlement_factor for d in future), 0.
        high = [y_max - sum(e.discounted_cash for e in events.values()
                           if e.ex_time > t and (e.ex_time < d.time or
                               (e.ex_time == d.time and d.phase == "after")))
                - option.strike*d.settlement_factor for d in future]
        return 0., max(0., max(high))

    mandatory = set(anchors)
    blocks, widths, weights, left, right = [], [], [], [], []
    for a, b in reversed(list(zip(times, times[1:]))):
        h = b-a
        steps = [((a+b)/2, h/2, 1.), (a, h/2, 1.)] if b in mandatory else [(a, h, .5)]
        for earlier, width, weight in steps:
            lower, upper = boundary(earlier)
            widths.append(width); weights.append(weight); left.append(lower); right.append(upper)
        if a in mandatory:
            arrays = tuple(np.asarray(v, dtype=float) for v in (widths, weights, left, right))
            for array in arrays:
                array.setflags(write=False)
            blocks.append(Block(float(a), *arrays))
            widths, weights, left, right = [], [], [], []
    grid.setflags(write=False); diffusion.setflags(write=False)
    return PreparedProblem(model, option, rights, events, grid, diffusion, tuple(blocks), len(times)-1)


def price_prepared(problem, advance=advance_scipy):
    """Price with financial event maps applied in Python between native blocks."""
    p = problem
    if p.model.volatility == 0:
        value = deterministic_value(p.model, p.option, p.rights)
    else:
        values = _apply(p.model, p.option, p.grid, np.zeros_like(p.grid),
                        p.option.maturity, p.rights, p.events)
        for b in p.blocks:
            values = advance(p.diffusion, values, b.widths, b.theta, b.left, b.right)
            values = _apply(p.model, p.option, p.grid, values, b.earlier, p.rights, p.events)
        value = float(np.interp(p.model.initial_state, p.grid, values))
    return _result(value, p.model, p.grid, p.time_intervals, p.rights, p.events)
