"""Surrogate testing helpers for causality metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable # Allow passing a function as an argument to run_surrogate_test

import numpy as np

# slots=True in the dataclass decorator makes the class more memory efficient by preventing the creation of a __dict__ for each instance, which can save memory when creating many instances of the class.
# not using frozen=True allows the attributes of the class to be mutable, so they can be modified after the instance is created. This is useful for storing results that may be computed or updated later.
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

    Shuffle tests destroy the temporal relationship between series_one and series_two to build a null distribution
    One-sided p-value is computed testing whether the real value is unusally HIGH compared to the null
    It is appropriate for causality metrics like TE, CCM, and Granger F-statistics, where a higher value indicates stronger evidence of causality.
    """
    if method not in ("shuffle", "bootstrap"):
        raise ValueError(f"Invalid surrogate method '{method}'. Must be 'shuffle' or 'bootstrap'.")

    rng = make_rng(seed)
    x = np.asarray(series_one)
    y = np.asarray(series_two)

    # real_func can be granger_causality, compute_ccm, compute_te, or any other function that takes two series and returns a float metric.
    # **kwargs allows passing additional keyword arguments to real_func, such as lag or embed_dim for CCM or TE.
    real_value = float(real_func(x, y, **kwargs))
    surrogate_values: list[float] = []
    n_failed = 0

    for _ in range(n_surrogates):
        if method == "shuffle":
            # reorder the elements of x randomly to create a surrogate series, while keeping y fixed. This breaks any temporal relationship between x and y.
            xs = rng.permutation(x)
        else:
            # bootstrap resampling: sample with replacement from x to create a surrogate series, while keeping y fixed. This preserves the distribution of x but breaks any temporal relationship between x and y.
            # key difference: shuffle permutes the original series, while bootstrap resamples with replacement, allowing for repeated values and potentially different lengths.
            indices = rng.integers(0, len(x), size=len(x))
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
    elif result.method == "bootstrap":
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
    else:
        print(f"Unknown surrogate method '{result.method}' for {metric_name} direction {direction}")
    if result.n_failed > 0:
        print(f"Warning: {result.n_failed} surrogate computations failed and were excluded from the results.")
    print("-" * 80)

