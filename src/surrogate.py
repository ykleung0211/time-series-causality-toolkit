"""Surrogate testing helpers for causality metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


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
    """Run a shuffle or bootstrap surrogate test.

    Shuffle tests are reported as p-values. Bootstrap tests are reported as percentile confidence intervals.
    """
    rng = make_rng(seed)
    x = np.asarray(series_one)
    y = np.asarray(series_two)
    real_value = float(real_func(x, y, **kwargs))
    surrogate_values: list[float] = []

    for _ in range(n_surrogates):
        if method == "shuffle":
            xs = rng.permutation(x)
        else:
            indices = rng.integers(0, len(x), size=len(x))
            xs = x[indices]
        try:
            surrogate_values.append(float(real_func(xs, y, **kwargs)))
        except Exception:
            surrogate_values.append(np.nan)

    values = np.asarray([value for value in surrogate_values if not np.isnan(value)], dtype=float)
    result = SurrogateResult(method=method, real=real_value, surrogates=values, seed=seed)

    if len(values) == 0:
        return result

    if method == "shuffle":
        result.p_value = float((np.sum(values >= real_value) + 1) / (len(values) + 1))
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
    if result.method == "shuffle":
        print(f"Shuffle surrogate test for {metric_name} direction {direction}")
        print("-" * 80)
        print(f"Real value:           {result.real:.6f}")
        print(f"p-value:              {result.p_value:.4f}" if result.p_value is not None else "p-value:              n/a")
        print(f"Surrogate mean:       {surrogate_mean:.6f}" if surrogate_mean is not None else "Surrogate mean:       n/a")
        print(f"Surrogate median:     {surrogate_median:.6f}" if surrogate_median is not None else "Surrogate median:     n/a")
        print(f"Surrogate std:        {surrogate_std:.6f}")
    else:
        print(f"Bootstrap confidence interval for {metric_name} direction {direction}")
        print("-" * 80)
        print(f"Real value:           {result.real:.6f}")
        if result.ci_low is not None and result.ci_high is not None:
            print(f"95% CI:               [{result.ci_low:.6f}, {result.ci_high:.6f}]")
        else:
            print("95% CI:               n/a")
        print(f"Surrogate mean:       {surrogate_mean:.6f}" if surrogate_mean is not None else "Surrogate mean:       n/a")
        print(f"Surrogate median:     {surrogate_median:.6f}" if surrogate_median is not None else "Surrogate median:     n/a")
        print(f"Surrogate std:        {surrogate_std:.6f}")
    print("-" * 80)
