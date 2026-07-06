# Preprocessing utilities for time-series analysis.

from __future__ import annotations

import numpy as np
import pandas as pd

from src.plotting import plot_line_trend


def smooth_series(series: pd.Series, window: int = 5) -> pd.Series:
    # Apply centered rolling-mean smoothing.
    return series.rolling(window=window, center=True).mean().dropna()


def downsample_series(series: pd.Series, step: int | None = None, freq: str | None = None) -> pd.Series:
    # Downsample by integer step or pandas frequency alias.
    if freq:
        return series.resample(freq).last().dropna()
    if step and step > 1:
        return series.iloc[::step]
    return series


def lagged_cross_correlation(series_one: pd.Series, series_two: pd.Series, max_lag: int = 30) -> pd.DataFrame:
    # Compute Pearson correlation across a symmetric lag window.
    common_index = series_one.index.intersection(series_two.index)
    left = pd.Series(series_one.loc[common_index]).dropna()
    right = pd.Series(series_two.loc[common_index]).dropna()

    rows: list[dict[str, float | int]] = []
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            aligned_left = left.iloc[:-lag]
            aligned_right = right.iloc[lag:]
        elif lag < 0:
            aligned_left = left.iloc[-lag:]
            aligned_right = right.iloc[:lag]
        else:
            aligned_left = left
            aligned_right = right

        if len(aligned_left) < 2 or len(aligned_right) < 2:
            correlation = np.nan
        else:
            correlation = float(aligned_left.corr(aligned_right))
        rows.append({"lag": lag, "correlation": correlation})

    return pd.DataFrame(rows)


def prompt_lagged_cross_correlation(
    series_one: pd.Series,
    series_two: pd.Series,
    label_one: str,
    label_two: str,
    max_lag: int = 30,
) -> pd.DataFrame | None:
    """Ask whether to plot lagged cross-correlation and return the computed frame."""
    answer = input(f"Do you want to plot lagged cross-correlation (LCC) for {label_one} vs {label_two}? [y/N]: ").strip().lower()
    if answer not in {"y", "yes"}:
        return None

    frame = lagged_cross_correlation(series_one, series_two, max_lag=max_lag)
    valid = frame["correlation"].dropna()
    if valid.empty:
        print(f"No valid LCC values were produced for {label_one} vs {label_two}.")
        return frame

    best_index = valid.abs().idxmax()
    best_row = frame.loc[best_index]
    print(f"Peak absolute LCC at lag {int(best_row['lag'])}: {best_row['correlation']:.6f}")
    plot_line_trend(
        frame,
        "lag",
        ["correlation"],
        f"Lagged cross-correlation: {label_one} vs {label_two}",
        "Lag",
        "Pearson correlation",
    )
    return frame
