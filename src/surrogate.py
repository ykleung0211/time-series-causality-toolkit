"""Surrogate testing helpers for causality metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# Methods that build a null distribution (p-value + effective score)
NULL_METHODS = ("shuffle", "circular_shift")
ALL_METHODS = ("shuffle", "circular_shift", "bootstrap")


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


def run_surrogate_test(
    real_func: Callable[..., float],
    series_one: np.ndarray,
    series_two: np.ndarray,
    n_surrogates: int = 200,
    method: str = "shuffle",
    seed: int | None = None,
    **kwargs,
) -> SurrogateResult:
    """Run a shuffle, circular-shift or bootstrap surrogate test.

    shuffle: random permutation of series_one. Destroys its own autocorrelation too.
    circular_shift: rotate series_one by a random amount. Keeps its autocorrelation
        and distribution, but breaks its alignment with series_two.
    bootstrap: i.i.d. resampling with replacement (a null-style band, not a true CI).

    shuffle and circular_shift return a one-sided p-value
    (count of surrogates >= real + 1) / (n + 1) and an effective score (real - mean surrogate).
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

    real_value = float(real_func(x, y, **kwargs))
    surrogate_values: list[float] = []
    n_failed = 0

    for _ in range(n_surrogates):
        if method == "shuffle":
            xs = rng.permutation(x)
        elif method == "circular_shift":
            min_shift = max(1, n // 10)
            shift = int(rng.integers(min_shift, n - min_shift + 1))
            xs = np.roll(x, shift)
        else:
            indices = rng.integers(0, n, size=n)
            xs = x[indices]
        try:
            surrogate_values.append(float(real_func(xs, y, **kwargs)))
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
        print(f"Surrogate mean: {surrogate_mean:.6f}" if surrogate_mean is not None else "Surrogate mean: n/a")
        print(f"Surrogate median: {surrogate_median:.6f}" if surrogate_median is not None else "Surrogate median: n/a")
        print(f"Surrogate std: {surrogate_std:.6f}")
    elif result.method == "bootstrap":
        print(f"Bootstrap interval for {metric_name} direction {direction}")
        print("-" * 80)
        print(f"Real value: {result.real:.6f}")
        if result.ci_low is not None and result.ci_high is not None:
            print(f"95% interval: [{result.ci_low:.6f}, {result.ci_high:.6f}]")
        else:
            print("95% interval: n/a")
        print(f"Surrogate mean: {surrogate_mean:.6f}" if surrogate_mean is not None else "Surrogate mean: n/a")
        print(f"Surrogate median: {surrogate_median:.6f}" if surrogate_median is not None else "Surrogate median: n/a")
        print(f"Surrogate std: {surrogate_std:.6f}")
    else:
        print(f"Unknown surrogate method '{result.method}' for {metric_name} direction {direction}")
    if result.n_failed > 0:
        print(f"Warning: {result.n_failed} surrogate computations failed and were excluded from the results.")
    print("-" * 80)