"""Causal analysis utilities for pairs of one-dimensional time series."""

# Defer type-hint evaluation on older python versions
from __future__ import annotations 

# Silence warnings from crossmapy, dtw and statsmodels libraries
from contextlib import redirect_stderr, redirect_stdout 
from io import StringIO
import logging
import warnings

# For parameters like embedding dimension and lag in sweep functions
from collections.abc import Iterable

import numpy as np
import pandas as pd
from crossmapy import ccm
from dtw import dtw
from infomeasure import transfer_entropy
from statsmodels.tsa.stattools import grangercausalitytests


def compute_dtw_sequence(series_one: pd.Series, series_two: pd.Series):
    """Compute DTW on the raw sequence order of two series, ignoring their index alignment.
    
    This is uesed in the DTW alignment mode for variable-lag style analysis.
    """

    x = np.asarray(series_one, dtype=float).reshape(-1, 1)
    y = np.asarray(series_two, dtype=float).reshape(-1, 1)
    if x.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("Input series must not be empty.")
    
    # keep_internals-True retains the cost matrix and warping path for later extraction
    return dtw(x, y, dist_method="euclidean", keep_internals=True)


def extract_warping_path(dtw_alignment) -> tuple[np.ndarray, np.ndarray]:
    """Return the DTW warping path as two integer index arrays."""
    try:
        # Attempt to access the path via the public attributes of the DTW object
        index_one = np.asarray(dtw_alignment.index1, dtype=int)
        index_two = np.asarray(dtw_alignment.index2, dtype=int)
        if index_one.size and index_two.size:
            return index_one, index_two
    except Exception:
        pass

    try:
        # Fallback to the internal path attribute if available
        # path is expected to be an array of shape (n_steps, 2) with integer indices (i, j)
        path = np.asarray(dtw_alignment.path, dtype=int)
    except Exception as exc:
        raise ValueError("DTW alignment does not expose a warping path.") from exc

    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("DTW warping path must be a two-column array of index pairs.")
    return path[:, 0], path[:, 1]


def warp_series_to_match(
    series_source: pd.Series,
    series_target: pd.Series,
    dtw_alignment,
) -> pd.Series:
    """Warp series_source onto the index of series_target using a DTW path.
    
    This variant ignores the original index alignment and uses the DTW warping path in sequence-order space.
    The returned series has the same index as series_target
    """

    if not isinstance(series_source, pd.Series) or not isinstance(series_target, pd.Series):
        raise TypeError("warp_series_to_match expects pandas Series inputs.")

    x = pd.to_numeric(pd.Series(series_source), errors="coerce").dropna().to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(series_target), errors="coerce").dropna().to_numpy(dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Input series must be one-dimensional.")
    if x.size == 0 or y.size == 0:
        raise ValueError("Input series must not be empty.")
    
    path_source, path_target = extract_warping_path(dtw_alignment)
    if path_source.size == 0 or path_target.size == 0:
        raise ValueError("DTW warping path is empty.")
    
    # Check that the DTW path indices are compatible with the raw sequence lengths
    # If x.shape[0] contains n elements, valid indices are in [0, n-1], so max index must be < n
    if int(path_source.max()) >= x.shape[0] or int(path_target.max()) >= y.shape[0]:
        raise ValueError("DTW path indices do not match the series lengths.")
    
    # Create an array to hold the warped values, initialized with NaN
    warped_values = np.empty_like(y, dtype=float)
    warped_values.fill(np.nan)

    # For each target index j
    for j in range(y.shape[0]):
        # Collect all source indices i that map to this target index j
        # path_target == j gives a boolean mask of where the target index matches j
        # then path_source[path_target == j] gives the corresponding source indices i
        # so matches is an array of source indices that map to target index j
        matches = path_source[path_target == j]

        # if y is shorter than x, then matches.size could be 0 or just 1
        if matches.size > 0:
            # x[matches] gives the source values that map to target index j
            # Average them to handle many-to-one mappings in the DTW path
            warped_values[j] = np.mean(x[matches])
        else:
            # If there is no exact match (j is not in path_target), use nearest neighbor in the DTW path
            # Find the index in path_target that is closest to j
            # np.argmin picks the first occurrence of the minimum distance
            nearest_idx = int(np.argmin(np.abs(path_target - j)))
            warped_values[j] = float(x[path_source[nearest_idx]])

    warped = pd.Series(warped_values, index=pd.Series(series_target).index, name=getattr(series_source, "name", None))
    return warped

def _prepare_granger_frame(series_target: pd.Series, series_source: pd.Series) -> pd.DataFrame:
    target = pd.to_numeric(pd.Series(series_target), errors="coerce").dropna().reset_index(drop=True)
    source = pd.to_numeric(pd.Series(series_source), errors="coerce").dropna().reset_index(drop=True)
    limit = min(len(target), len(source))
    if limit < 2:
        return pd.DataFrame(columns=["target", "source"])
    frame = pd.DataFrame({"target": target.iloc[:limit].to_numpy(), "source": source.iloc[:limit].to_numpy()})
    return frame.dropna()


def _granger_direction_report(series_target: pd.Series, series_source: pd.Series, target_name: str, source_name: str, maxlag: int) -> pd.DataFrame:
    frame = _prepare_granger_frame(series_target, series_source)
    rows: list[dict[str, float | int | str]] = []
    
    # If the series are too short for the specified maxlag, return an empty DataFrame with the expected columns
    # Degree of freedom must be greater than 0
    # Maxlag of data used maxlag number of degrees of freedom
    # Intercept uses 1 degree of freedom, so we need at least maxlag + 2 observations to run the test
    if len(frame) < maxlag + 2:
        return pd.DataFrame(columns=["direction", "lag", "f_stat", "p_value"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            result = grangercausalitytests(frame[["target", "source"]], maxlag=maxlag, verbose=False)

    for lag, outputs in result.items():
        # outputs is a tuple of (test_statistic, p_value, df_denom, df_num)
        # ssr_ftest returns a tuple of (F-statistic, p-value, df_denom, df_num)
        statistic = outputs[0].get("ssr_ftest")
        if statistic is None:
            continue
        rows.append(
            {
                "direction": f"{source_name} -> {target_name}",
                "lag": int(lag),
                "f_stat": float(statistic[0]),
                "p_value": float(statistic[1]),
            }
        )

    return pd.DataFrame(rows)


def run_granger_causality_report(
    series_one: pd.Series,
    series_two: pd.Series,
    name_one: str = "Series 1",
    name_two: str = "Series 2",
    maxlag: int = 5,
    verbose: bool = False,
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

    if verbose:
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
    """Return the max F-statistics for both Granger directions."""
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

    # for each lag output
    for outputs in result.values():
        # outputs is a tuple of (test_statistic, p_value, df_denom, df_num)
        # ssr_ftest returns a tuple of (F-statistic, p-value, df_denom, df_num)
        statistic = outputs[0].get("ssr_ftest")
        if statistic is None:
            continue
        best = max(best, float(statistic[0]))
    return best


def compute_te(series_one: np.ndarray | pd.Series, series_two: np.ndarray | pd.Series, embed_dim: int, lag: int) -> float:
    """Compute ordinal transfer entropy from series_one to series_two."""
    
    if embed_dim < 1 or lag < 1:
        raise ValueError(f"Embedding dimension and lag must be positive integers. Got: embed_dim={embed_dim}, lag={lag}")
    
    # Suppress logging from the infomeasure library to avoid cluttering output
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.ERROR)
    try:
        value = transfer_entropy(
            np.asarray(series_one),
            np.asarray(series_two),
            approach="ordinal",
            embedding_dim=int(embed_dim),
            step_size=int(lag),
        )
    finally:
        # Restore the previous logging level to avoid affecting other parts of the program
        logging.disable(previous_disable_level)
    return float(value)


def compute_ccm(series_one: np.ndarray | pd.Series, series_two: np.ndarray | pd.Series, embed_dim: int, lag: int) -> tuple[float, float]:
    """Compute CCM directional scores for both directions."""
    
    if embed_dim < 1 or lag < 1:
        raise ValueError(f"Embedding dimension and lag must be positive integers. Got: embed_dim={embed_dim}, lag={lag}")

    # Convert input series to a 2D array for crossmapy and define the CCM model with specified embedding dimension and lag
    data = np.column_stack([np.asarray(series_one), np.asarray(series_two)])
    model = ccm.ConvergeCrossMapping(embed_dim=int(embed_dim), lag=max(1, int(lag)))
    
    # Suppress warnings from crossmapy and related libraries during fitting
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in sqrt")
        warnings.filterwarnings("ignore", message=".*does not have enough neighbors.*")
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            model.fit(data)
    scores = model.scores
    # Scores is a 2x2 array
    # scores[0, 1] is the score for series_one -> series_two
    # scores[1, 0] is for series_two -> series_one
    # example: scores = [[0.0, 0.5], [0.6, 0.0]]
    # series_one -> series_two = 0.6, series_two -> series_one = 0.5
    return float(scores[0, 1]), float(scores[1, 0])


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
    """Evaluate TE for all lags in [1, max_lag] and embedding dimensions in [2, max_embed_dim]."""
    return sweep_transfer_entropy(series_one, series_two, range(2, max_embed_dim + 1), range(1, max_lag + 1))


def sweep_ccm(
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


def sweep_ccm_grid(
    series_one: np.ndarray | pd.Series,
    series_two: np.ndarray | pd.Series,
    max_lag: int,
    max_embed_dim: int,
) -> pd.DataFrame:
    """Evaluate CCM for all lags in [1, max_lag] and embedding dimensions in [2, max_embed_dim]."""
    return sweep_ccm(series_one, series_two, range(2, max_embed_dim + 1), range(1, max_lag + 1))


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
            # current is the current library fraction, starting from library_step and incrementing by library_step until 1.0
            current = float(library_step)
            
            # Use a small epsilon to avoid floating-point precision issues when comparing to 1.0
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


def print_parameter_sweep_report(
    frame: pd.DataFrame,
    metric_name: str,
    label_one: str,
    label_two: str,
    verbose: bool = False,
) -> dict[str, object]:
    """Print a text report for a TE or CCM parameter grid and return the best rows.

    The grid is expected to contain ``embed_dim``, ``lag``, ``one_two``, and
    ``two_one`` columns.
    """
    summary: dict[str, object] = {
        "one_to_two": None,
        "two_to_one": None,
        "stronger_direction": None,
    }
    if frame.empty:
        if verbose:
            print("\n" + "=" * 80)
            print(f"{metric_name.upper()} PARAMETER SWEEP REPORT")
            print("=" * 80)
            print("No valid rows were produced for the parameter sweep.")
        return summary

    for column, direction_label in (("one_two", f"{label_one} -> {label_two}"), ("two_one", f"{label_two} -> {label_one}")):
        direction_frame = frame[["embed_dim", "lag", column]].dropna().copy()
        if direction_frame.empty:
            if verbose:
                print(f"\nDirection: {direction_label}")
                print("No valid scores for this direction.")
            continue

        best_index = direction_frame[column].idxmax()
        best_row = direction_frame.loc[best_index]
        summary_key = "one_to_two" if column == "one_two" else "two_to_one"
        summary[summary_key] = best_row.to_dict()

        if verbose:
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
        if verbose:
            print(f"\nOverall stronger direction by best score: {stronger_direction}")

    return summary
