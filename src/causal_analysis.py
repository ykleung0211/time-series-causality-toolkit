"""Causal analysis utilities for pairs of one-dimensional time series."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import logging
import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd
from crossmapy import ccm
from dtw import dtw
from infomeasure import transfer_entropy
from statsmodels.tsa.stattools import grangercausalitytests


def compute_dtw(series_one: pd.Series, series_two: pd.Series):
    """Compute a DTW alignment object for two aligned series."""
    common_index = series_one.index.intersection(series_two.index)
    left = series_one.loc[common_index].values.reshape(-1, 1)
    right = series_two.loc[common_index].values.reshape(-1, 1)
    return dtw(left, right, dist_method="euclidean", keep_internals=True)


def _prepare_granger_frame(series_target: pd.Series, series_source: pd.Series) -> pd.DataFrame:
    common_index = series_target.index.intersection(series_source.index)
    frame = pd.DataFrame({"target": series_target.loc[common_index], "source": series_source.loc[common_index]})
    return frame.dropna()


def _granger_direction_report(series_target: pd.Series, series_source: pd.Series, target_name: str, source_name: str, maxlag: int) -> pd.DataFrame:
    frame = _prepare_granger_frame(series_target, series_source)
    rows: list[dict[str, float | int | str]] = []
    if len(frame) < maxlag + 2:
        return pd.DataFrame(columns=["direction", "lag", "f_stat", "p_value", "ssr_ftest_pvalue"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            result = grangercausalitytests(frame[["target", "source"]], maxlag=maxlag, verbose=False)

    for lag, outputs in result.items():
        statistic = outputs[0].get("ssr_ftest")
        if statistic is None:
            continue
        rows.append(
            {
                "direction": f"{source_name} -> {target_name}",
                "lag": int(lag),
                "f_stat": float(statistic[0]),
                "p_value": float(statistic[1]),
                "ssr_ftest_pvalue": float(statistic[1]),
            }
        )

    return pd.DataFrame(rows)


def run_granger_causality_report(
    series_one: pd.Series,
    series_two: pd.Series,
    name_one: str = "Series 1",
    name_two: str = "Series 2",
    maxlag: int = 5,
) -> dict[str, pd.DataFrame | dict[str, object]]:
    """Run Granger causality tests in both directions and return structured results."""
    one_to_two = _granger_direction_report(series_two, series_one, name_two, name_one, maxlag)
    two_to_one = _granger_direction_report(series_one, series_two, name_one, name_two, maxlag)

    def _best_row(frame: pd.DataFrame) -> pd.Series | None:
        if frame.empty or frame["f_stat"].dropna().empty:
            return None
        return frame.loc[frame["f_stat"].idxmax()]

    best_one_two = _best_row(one_to_two)
    best_two_one = _best_row(two_to_one)

    print("\n" + "=" * 80)
    print("GRANGER CAUSALITY REPORT")
    print("=" * 80)
    for frame in (one_to_two, two_to_one):
        if frame.empty:
            print(f"No valid Granger result for {frame.attrs.get('direction', 'an unknown direction')}")
            continue
        direction = frame.iloc[0]["direction"]
        print(f"\nDirection: {direction}")
        print(frame.to_string(index=False))

    print("\nSummary:")
    if best_one_two is not None:
        print(
            f"{name_one} -> {name_two}: strongest evidence at lag {int(best_one_two['lag'])} "
            f"(F={best_one_two['f_stat']:.6f}, p={best_one_two['p_value']:.6g})"
        )
    else:
        print(f"{name_one} -> {name_two}: no valid result")
    if best_two_one is not None:
        print(
            f"{name_two} -> {name_one}: strongest evidence at lag {int(best_two_one['lag'])} "
            f"(F={best_two_one['f_stat']:.6f}, p={best_two_one['p_value']:.6g})"
        )
    else:
        print(f"{name_two} -> {name_one}: no valid result")

    stronger_direction = None
    if best_one_two is not None and best_two_one is not None:
        stronger_direction = f"{name_one} -> {name_two}" if best_one_two["f_stat"] >= best_two_one["f_stat"] else f"{name_two} -> {name_one}"

    if stronger_direction:
        print(f"\nOverall stronger direction by max F-stat: {stronger_direction}")

    return {
        "one_to_two": one_to_two,
        "two_to_one": two_to_one,
        "summary": {
            "best_one_to_two": None if best_one_two is None else best_one_two.to_dict(),
            "best_two_to_one": None if best_two_one is None else best_two_one.to_dict(),
            "stronger_direction": stronger_direction,
        },
    }


def granger_strength(series_one: pd.Series, series_two: pd.Series, maxlag: int, verbose: bool = False) -> tuple[float, float]:
    """Compatibility wrapper returning max F-statistics for both Granger directions."""
    report = run_granger_causality_report(series_one, series_two, maxlag=maxlag)
    one_to_two = report["one_to_two"]
    two_to_one = report["two_to_one"]
    max_one_two = float(one_to_two["f_stat"].max()) if not one_to_two.empty else 0.0
    max_two_one = float(two_to_one["f_stat"].max()) if not two_to_one.empty else 0.0
    return max_one_two, max_two_one


def granger_direction_score(series_target: pd.Series, series_source: pd.Series, maxlag: int) -> float:
    """Return the strongest Granger F-statistic for a single source-to-target direction."""
    frame = _prepare_granger_frame(series_target, series_source)
    if len(frame) < maxlag + 2:
        return float(0.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            result = grangercausalitytests(frame[["target", "source"]], maxlag=maxlag, verbose=False)

    best = 0.0
    for outputs in result.values():
        statistic = outputs[0].get("ssr_ftest")
        if statistic is None:
            continue
        best = max(best, float(statistic[0]))
    return best


def compute_te(series_one: np.ndarray | pd.Series, series_two: np.ndarray | pd.Series, embed_dim: int, lag: int) -> float:
    """Compute ordinal transfer entropy from series_one to series_two."""
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.ERROR)
    try:
        value = transfer_entropy(
            np.asarray(series_one),
            np.asarray(series_two),
            approach="ordinal",
            embedding_dim=max(1, int(embed_dim)),
            step_size=max(1, int(lag)),
        )
    finally:
        logging.disable(previous_disable_level)
    return float(value)


def compute_ccm(series_one: np.ndarray | pd.Series, series_two: np.ndarray | pd.Series, embed_dim: int, lag: int) -> tuple[float, float]:
    """Compute CCM directional scores for both directions."""
    data = np.column_stack([np.asarray(series_one), np.asarray(series_two)])
    model = ccm.ConvergeCrossMapping(embed_dim=max(1, int(embed_dim)), lag=max(1, int(lag)))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in sqrt")
        model.fit(data)
    scores = model.scores
    return float(scores[1, 0]), float(scores[0, 1])


def sweep_transfer_entropy(
    series_one: np.ndarray | pd.Series,
    series_two: np.ndarray | pd.Series,
    embed_dims: Iterable[int],
    lags: Iterable[int],
) -> pd.DataFrame:
    """Evaluate TE over a grid of embedding dimensions and lags."""
    rows: list[dict[str, float | int]] = []
    for embed_dim in embed_dims:
        for lag in lags:
            try:
                one_two = compute_te(series_one, series_two, embed_dim, lag)
                two_one = compute_te(series_two, series_one, embed_dim, lag)
                rows.append({"embed_dim": int(embed_dim), "lag": int(lag), "one_two": one_two, "two_one": two_one})
            except Exception:
                rows.append({"embed_dim": int(embed_dim), "lag": int(lag), "one_two": np.nan, "two_one": np.nan})
    return pd.DataFrame(rows)


def sweep_transfer_entropy_grid(
    series_one: np.ndarray | pd.Series,
    series_two: np.ndarray | pd.Series,
    max_lag: int,
    max_embed_dim: int,
) -> pd.DataFrame:
    """Evaluate TE for all lags in [0, max_lag] and embedding dimensions in [0, max_embed_dim]."""
    return sweep_transfer_entropy(series_one, series_two, range(0, max_embed_dim + 1), range(0, max_lag + 1))


def sweep_ccm_grid(
    series_one: np.ndarray | pd.Series,
    series_two: np.ndarray | pd.Series,
    embed_dims: Iterable[int],
    lags: Iterable[int],
) -> pd.DataFrame:
    """Evaluate CCM over a grid of embedding dimensions and lags."""
    rows: list[dict[str, float | int]] = []
    for embed_dim in embed_dims:
        for lag in lags:
            try:
                one_two, two_one = compute_ccm(series_one, series_two, embed_dim, lag)
                rows.append({"embed_dim": int(embed_dim), "lag": int(lag), "one_two": one_two, "two_one": two_one})
            except Exception:
                rows.append({"embed_dim": int(embed_dim), "lag": int(lag), "one_two": np.nan, "two_one": np.nan})
    return pd.DataFrame(rows)


def sweep_ccm_grid_all(
    series_one: np.ndarray | pd.Series,
    series_two: np.ndarray | pd.Series,
    max_lag: int,
    max_embed_dim: int,
) -> pd.DataFrame:
    """Evaluate CCM for all lags in [0, max_lag] and embedding dimensions in [0, max_embed_dim]."""
    return sweep_ccm_grid(series_one, series_two, range(0, max_embed_dim + 1), range(0, max_lag + 1))


def sweep_ccm_convergence(
    series_one: np.ndarray | pd.Series,
    series_two: np.ndarray | pd.Series,
    embed_dim: int,
    lag: int,
    library_fractions: Iterable[float] | None = None,
    library_step: float | None = None,
) -> pd.DataFrame:
    """Approximate CCM convergence by increasing the library size."""
    if library_fractions is None:
        if library_step is None:
            library_fractions = (0.2, 0.4, 0.6, 0.8, 1.0)
        else:
            fractions: list[float] = []
            current = float(library_step)
            while current <= 1.0 + 1e-9:
                fractions.append(round(current, 10))
                current += float(library_step)
            library_fractions = fractions

    x = np.asarray(series_one)
    y = np.asarray(series_two)
    n_obs = min(len(x), len(y))
    rows: list[dict[str, float | int]] = []

    for fraction in library_fractions:
        n_keep = max(10, int(n_obs * float(fraction)))
        prefix_x = x[:n_keep]
        prefix_y = y[:n_keep]
        try:
            one_two, two_one = compute_ccm(prefix_x, prefix_y, embed_dim, lag)
        except Exception:
            one_two, two_one = np.nan, np.nan
        rows.append({"fraction": float(fraction), "n_obs": n_keep, "one_two": one_two, "two_one": two_one})

    return pd.DataFrame(rows)


def sweep_ccm_convergence_steps(
    series_one: np.ndarray | pd.Series,
    series_two: np.ndarray | pd.Series,
    embed_dim: int,
    lag: int,
    library_step: float,
) -> pd.DataFrame:
    """Convenience wrapper for CCM convergence using a user-provided step."""
    return sweep_ccm_convergence(series_one, series_two, embed_dim, lag, library_step=library_step)


def print_parameter_sweep_report(frame: pd.DataFrame, metric_name: str, label_one: str, label_two: str) -> dict[str, object]:
    """Print a text report for a TE or CCM parameter grid and return the best rows.

    The grid is expected to contain ``embed_dim``, ``lag``, ``one_two``, and
    ``two_one`` columns.
    """
    print("\n" + "=" * 80)
    print(f"{metric_name.upper()} PARAMETER SWEEP REPORT")
    print("=" * 80)

    summary: dict[str, object] = {
        "one_to_two": None,
        "two_to_one": None,
        "stronger_direction": None,
    }
    if frame.empty:
        print("No valid rows were produced for the parameter sweep.")
        return summary

    for column, direction_label in (("one_two", f"{label_one} -> {label_two}"), ("two_one", f"{label_two} -> {label_one}")):
        direction_frame = frame[["embed_dim", "lag", column]].dropna().copy()
        if direction_frame.empty:
            print(f"\nDirection: {direction_label}")
            print("No valid scores for this direction.")
            continue

        best_index = direction_frame[column].idxmax()
        best_row = direction_frame.loc[best_index]
        summary_key = "one_to_two" if column == "one_two" else "two_to_one"
        summary[summary_key] = best_row.to_dict()

        print(f"\nDirection: {direction_label}")
        print(direction_frame.to_string(index=False))
        print(
            f"Best combination: lag={int(best_row['lag'])}, embed_dim={int(best_row['embed_dim'])}, "
            f"score={float(best_row[column]):.6f}"
        )

    best_one = summary["one_to_two"]
    best_two = summary["two_to_one"]
    if isinstance(best_one, dict) and isinstance(best_two, dict):
        stronger_direction = f"{label_one} -> {label_two}" if float(best_one.get("one_two", float("-inf"))) >= float(best_two.get("two_one", float("-inf"))) else f"{label_two} -> {label_one}"
        summary["stronger_direction"] = stronger_direction
        print(f"\nOverall stronger direction by best score: {stronger_direction}")

    return summary
