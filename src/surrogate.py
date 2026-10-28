"""Surrogate testing helpers for causality metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# Null-distribution methods: p-value and effective score
NULL_METHODS = ("shuffle", "circular_shift")
# Resampling methods: percentile interval
INTERVAL_METHODS = ("block_bootstrap", "bootstrap")
ALL_METHODS = ("shuffle", "circular_shift", "block_bootstrap", "bootstrap")


@dataclass(slots=True)
class SurrogateResult:
    """Container for surrogate test output."""

    method: str
    real: float
    surrogates: np.ndarray
    seed: int | None = None
    p_value: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    n_failed: int = 0
    effective: float | None = None  # real value minus mean surrogate value


def make_rng(seed: int | None) -> np.random.Generator:
    """Create a reproducible NumPy random generator."""
    return np.random.default_rng(seed)


def _block_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap indices of length n."""
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n - block_len + 1, size=n_blocks)
    return np.concatenate([np.arange(s, s + block_len) for s in starts])[:n]


def run_surrogate_test(
    real_func: Callable[..., float],
    series_one: np.ndarray,
    series_two: np.ndarray,
    n_surrogates: int = 200,
    method: str = "shuffle",
    seed: int | None = None,
    block_len: int | None = None,
    **kwargs,
) -> SurrogateResult:
    """Run a surrogate or bootstrap test.

    shuffle: random permutation of series_one (destroys its own autocorrelation too).
    circular_shift: rotate series_one by a random amount (keeps its autocorrelation,
        breaks its alignment with series_two).
    block_bootstrap: resample blocks of BOTH series with the same indices. Keeps temporal
        and cross dependence. The percentile interval is an interval for the estimator.
        block_len defaults to about n ** (1/3).
    bootstrap: i.i.d. resampling of series_one only. This is a null-style band, not a
        confidence interval. Kept for backwards compatibility.

    shuffle and circular_shift return a one-sided p-value (count of surrogates >= real, plus 1)
    divided by (n + 1), and an effective score (real minus mean surrogate).
    The smallest attainable p-value is 1 / (n_surrogates + 1).
    """
    if method not in ALL_METHODS:
        raise ValueError(f"Invalid surrogate method '{method}'. Must be one of {ALL_METHODS}.")

    rng = make_rng(seed)
    x = np.asarray(series_one)
    y = np.asarray(series_two)
    n = len(x)
    if method == "circular_shift" and n < 20:
        raise ValueError("Circular-shift surrogates need at least 20 observations.")
    if method == "block_bootstrap":
        if len(y) != n:
            raise ValueError("block_bootstrap needs two series of equal length.")
        effective_block = block_len if block_len is not None else max(2, int(round(n ** (1.0 / 3.0))))
        effective_block = int(min(max(2, effective_block), n))

    real_value = float(real_func(x, y, **kwargs))
    surrogate_values: list[float] = []
    n_failed = 0

    for _ in range(n_surrogates):
        ys = y
        if method == "shuffle":
            xs = rng.permutation(x)
        elif method == "circular_shift":
            min_shift = max(1, n // 10)
            shift = int(rng.integers(min_shift, n - min_shift + 1))
            xs = np.roll(x, shift)
        elif method == "block_bootstrap":
            idx = _block_indices(n, effective_block, rng)
            xs = x[idx]
            ys = y[idx]
        else:
            indices = rng.integers(0, n, size=n)
            xs = x[indices]
        try:
            surrogate_values.append(float(real_func(xs, ys, **kwargs)))
        except Exception:
            n_failed += 1
            surrogate_values.append(np.nan)

    values = np.asarray([value for value in surrogate_values if not np.isnan(value)], dtype=float)
    result = SurrogateResult(method=method, real=real_value, surrogates=values, seed=seed)

    if n_failed > 0:
        print(f"Warning: {n_failed} surrogate computations failed and were excluded from the results.")
        result.n_failed = n_failed

    if len(values) == 0:
        return result

    if method in NULL_METHODS:
        result.p_value = float((np.sum(values >= real_value) + 1) / (len(values) + 1))
        result.effective = float(real_value - np.mean(values))
    else:
        result.ci_low = float(np.percentile(values, 2.5))
        result.ci_high = float(np.percentile(values, 97.5))

    return result


def print_surrogate_summary(metric_name: str, direction: str, result: SurrogateResult, verbose: bool = False) -> None:
    """Print the most relevant surrogate output for the selected method."""
    if not verbose:
        return
    values = result.surrogates
    surrogate_mean = float(np.mean(values)) if len(values) else None
    surrogate_median = float(np.median(values)) if len(values) else None
    surrogate_std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    print("\n" + "-" * 80)
    if result.method in NULL_METHODS:
        title = "Shuffle" if result.method == "shuffle" else "Circular-shift"
        print(f"{title} surrogate test for {metric_name} direction {direction}")
        print("-" * 80)
        print(f"Real value: {result.real:.6f}")
        print(f"p-value: {result.p_value:.4f}" if result.p_value is not None else "p-value: n/a")
        print(f"Effective score (real - surrogate mean): {result.effective:.6f}" if result.effective is not None else "Effective score: n/a")
    elif result.method in INTERVAL_METHODS:
        if result.method == "block_bootstrap":
            print(f"Block-bootstrap interval for {metric_name} direction {direction}")
        else:
            print(f"i.i.d. bootstrap null band (not a confidence interval) for {metric_name} direction {direction}")
        print("-" * 80)
        print(f"Real value: {result.real:.6f}")
        if result.ci_low is not None and result.ci_high is not None:
            print(f"95% interval: [{result.ci_low:.6f}, {result.ci_high:.6f}]")
        else:
            print("95% interval: n/a")
    else:
        print(f"Unknown surrogate method '{result.method}' for {metric_name} direction {direction}")
        print("-" * 80)
    print(f"Surrogate mean: {surrogate_mean:.6f}" if surrogate_mean is not None else "Surrogate mean: n/a")
    print(f"Surrogate median: {surrogate_median:.6f}" if surrogate_median is not None else "Surrogate median: n/a")
    print(f"Surrogate std: {surrogate_std:.6f}")
    if result.n_failed > 0:
        print(f"Warning: {result.n_failed} surrogate computations failed and were excluded from the results.")
    print("-" * 80)