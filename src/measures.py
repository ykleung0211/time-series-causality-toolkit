# Core causality and similarity metrics.

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import logging
import warnings
from collections.abc import Iterable
from io import StringIO

import numpy as np
import pandas as pd
from crossmapy import ccm
from dtw import dtw
from infomeasure import transfer_entropy
from statsmodels.tsa.stattools import grangercausalitytests


def compute_dtw(series_one: pd.Series, series_two: pd.Series):
    # Compute a DTW alignment object for two aligned series.
    common_index = series_one.index.intersection(series_two.index)
    left = series_one.loc[common_index].values.reshape(-1, 1)
    right = series_two.loc[common_index].values.reshape(-1, 1)
    return dtw(left, right, dist_method="euclidean", keep_internals=True)


def prepare_granger_df(series_one: pd.Series, series_two: pd.Series, name_one: str, name_two: str) -> pd.DataFrame:
    # Build a two-column frame suitable for granger causality tests.
    common_index = series_one.index.intersection(series_two.index)
    return pd.DataFrame(
        {
            f"{name_one}_log_return": series_one.loc[common_index],
            f"{name_two}_log_return": series_two.loc[common_index],
        }
    ).dropna()


def granger_strength(series_one: pd.Series, series_two: pd.Series, maxlag: int, verbose: bool = False) -> tuple[float, float]:
    # Return max F-statistics for both Granger directions.
    frame = pd.concat([series_one, series_two], axis=1).dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            result_one_two = grangercausalitytests(frame[[frame.columns[1], frame.columns[0]]], maxlag=maxlag, verbose=False)
            result_two_one = grangercausalitytests(frame[[frame.columns[0], frame.columns[1]]], maxlag=maxlag, verbose=False)

    def max_f(result: dict[int, tuple[dict, dict]]) -> float:
        values: list[float] = []
        for _, output in result.items():
            statistic = output[0].get("ssr_ftest")
            if statistic:
                values.append(float(statistic[0]))
        return max(values) if values else 0.0

    return max_f(result_one_two), max_f(result_two_one)


def compute_te(series_one: np.ndarray | pd.Series, series_two: np.ndarray | pd.Series, embed_dim: int, lag: int) -> float:
    # Compute ordinal transfer entropy from series_one to series_two.
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.ERROR)
    try:
        value = transfer_entropy(
            np.asarray(series_one),
            np.asarray(series_two),
            approach="ordinal",
            embedding_dim=embed_dim,
            step_size=lag,
        )
    finally:
        logging.disable(previous_disable_level)
    return float(value)


def compute_ccm(series_one: np.ndarray | pd.Series, series_two: np.ndarray | pd.Series, embed_dim: int, lag: int) -> tuple[float, float]:
    # Compute CCM directional scores for both directions.
    data = np.column_stack([np.asarray(series_one), np.asarray(series_two)])
    model = ccm.ConvergeCrossMapping(embed_dim=embed_dim, lag=lag)
    model.fit(data)
    scores = model.scores
    return float(scores[1, 0]), float(scores[0, 1])


def sweep_transfer_entropy(
    series_one: np.ndarray | pd.Series,
    series_two: np.ndarray | pd.Series,
    embed_dims: Iterable[int],
    lags: Iterable[int],
) -> pd.DataFrame:
    # Evaluate TE over a grid of embedding dimensions and lags.
    rows: list[dict[str, float | int]] = []
    for embed_dim in embed_dims:
        for lag in lags:
            try:
                one_two = compute_te(series_one, series_two, embed_dim, lag)
                two_one = compute_te(series_two, series_one, embed_dim, lag)
                rows.append({"embed_dim": embed_dim, "lag": lag, "one_two": one_two, "two_one": two_one})
            except Exception:
                rows.append({"embed_dim": embed_dim, "lag": lag, "one_two": np.nan, "two_one": np.nan})
    return pd.DataFrame(rows)


def sweep_ccm_grid(
    series_one: np.ndarray | pd.Series,
    series_two: np.ndarray | pd.Series,
    embed_dims: Iterable[int],
    lags: Iterable[int],
) -> pd.DataFrame:
    # Evaluate CCM over a grid of embedding dimensions and lags.
    rows: list[dict[str, float | int]] = []
    for embed_dim in embed_dims:
        for lag in lags:
            try:
                one_two, two_one = compute_ccm(series_one, series_two, embed_dim, lag)
                rows.append({"embed_dim": embed_dim, "lag": lag, "one_two": one_two, "two_one": two_one})
            except Exception:
                rows.append({"embed_dim": embed_dim, "lag": lag, "one_two": np.nan, "two_one": np.nan})
    return pd.DataFrame(rows)


def sweep_ccm_convergence(
    series_one: np.ndarray | pd.Series,
    series_two: np.ndarray | pd.Series,
    embed_dim: int,
    lag: int,
    library_fractions: Iterable[float] | None = None,
) -> pd.DataFrame:
    # Approximate CCM convergence by increasing the library size in prefixes.
    if library_fractions is None:
        library_fractions = (0.2, 0.4, 0.6, 0.8, 1.0)

    x = np.asarray(series_one)
    y = np.asarray(series_two)
    n_obs = min(len(x), len(y))
    rows: list[dict[str, float | int]] = []

    for fraction in library_fractions:
        n_keep = max(10, int(n_obs * fraction))
        prefix_x = x[:n_keep]
        prefix_y = y[:n_keep]
        try:
            one_two, two_one = compute_ccm(prefix_x, prefix_y, embed_dim, lag)
        except Exception:
            one_two, two_one = np.nan, np.nan
        rows.append({"fraction": float(fraction), "n_obs": n_keep, "one_two": one_two, "two_one": two_one})

    return pd.DataFrame(rows)
