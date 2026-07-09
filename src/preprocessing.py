"""Preprocessing utilities for time-series analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.plotting import plot_line_trend, plot_preprocessing_results, plot_single_series


def compute_returns(series: pd.Series) -> pd.Series:
    """Compute simple percentage returns for a numeric series."""
    cleaned = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    return cleaned.pct_change().dropna()


def compute_log_returns(series: pd.Series) -> pd.Series:
    """Compute log returns for a strictly positive numeric series."""
    cleaned = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    positive = cleaned[cleaned > 0]
    return np.log(positive).diff().dropna()


def smooth_series(series: pd.Series, window: int = 5) -> pd.Series:
    """Apply centered rolling-mean smoothing."""
    return pd.Series(series).rolling(window=window, center=True).mean().dropna()


def standardize_series(series: pd.Series) -> pd.Series:
    """Z-score standardize a series by subtracting the mean and dividing by the standard deviation."""
    cleaned = pd.Series(series).dropna()
    std = float(cleaned.std())
    if std == 0.0:
        return cleaned - cleaned.mean()
    return (cleaned - cleaned.mean()) / std


def downsample_series(series: pd.Series, step: int | None = None, freq: str | None = None) -> pd.Series:
    """Downsample by integer step or pandas frequency alias."""
    series = pd.Series(series)
    if freq:
        if not isinstance(series.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            raise TypeError("Frequency-based downsampling requires a DatetimeIndex or PeriodIndex.")
        return series.resample(freq).last().dropna()
    if step and step > 1:
        return series.iloc[::step].dropna()
    return series.dropna()


def lagged_cross_correlation(series_one: pd.Series, series_two: pd.Series, max_lag: int = 30) -> pd.DataFrame:
    """Compute Pearson correlation across a symmetric lag window."""
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


def summarize_preprocessing(before: pd.Series, after: pd.Series, before_label: str, after_label: str) -> dict[str, object]:
    """Return a small summary that notebooks can print or inspect."""
    return {
        "before_label": before_label,
        "after_label": after_label,
        "before_length": int(len(pd.Series(before).dropna())),
        "after_length": int(len(pd.Series(after).dropna())),
        "before_mean": float(pd.Series(before).dropna().mean()) if len(pd.Series(before).dropna()) else np.nan,
        "after_mean": float(pd.Series(after).dropna().mean()) if len(pd.Series(after).dropna()) else np.nan,
    }


def _prompt_bool(question: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    answer = input(f"{question} {suffix}: ").strip().lower()
    if not answer:
        return default
    return answer in {"y", "yes"}


def _prompt_choice(question: str, choices: list[str], default_index: int = 0) -> int:
    print(question)
    for index, choice in enumerate(choices, start=1):
        default_suffix = " (default)" if index - 1 == default_index else ""
        print(f"  {index}. {choice}{default_suffix}")
    answer = input(f"Select 1-{len(choices)} [{default_index + 1}]: ").strip()
    if not answer:
        return default_index
    try:
        selected = int(answer) - 1
    except ValueError:
        return default_index
    return selected if 0 <= selected < len(choices) else default_index


def _prompt_int(question: str, default: int) -> int:
    answer = input(f"{question} [{default}]: ").strip()
    if not answer:
        return default
    try:
        return int(answer)
    except ValueError:
        print(f"Invalid input. Using default {default}.")
        return default


def run_preprocessing_flow(
    series_one: pd.Series,
    series_two: pd.Series,
    label_one: str,
    label_two: str,
) -> tuple[pd.Series, pd.Series, dict[str, object], dict[str, object]]:
    """Interactively choose preprocessing steps for two series.

    The flow keeps the transformation names explicit so downstream labels clearly
    show whether the data are raw values, percentage changes, log changes,
    smoothed, downsampled, or z-scored.
    """
    if _prompt_bool(f"Plot the raw series for {label_one} and {label_two} on separate charts?", default=True):
        plot_single_series(series_one, f"Raw series: {label_one}", label_one)
        plot_single_series(series_two, f"Raw series: {label_two}", label_two)

    base_choice = _prompt_choice(
        "Choose the base representation for both series",
        ["raw values", "percentage changes", "log changes"],
        default_index=2,
    )
    if base_choice == 0:
        processed_one = pd.Series(series_one).dropna()
        processed_two = pd.Series(series_two).dropna()
        steps = ["raw values"]
    elif base_choice == 1:
        processed_one = compute_returns(series_one)
        processed_two = compute_returns(series_two)
        steps = ["percentage changes"]
    else:
        processed_one = compute_log_returns(series_one)
        processed_two = compute_log_returns(series_two)
        steps = ["log changes"]

    if _prompt_bool("Apply smoothing before the causal analysis?", default=False):
        window = _prompt_int("Smoothing window size", 5)
        before_one = processed_one
        before_two = processed_two
        processed_one = smooth_series(processed_one, window=window)
        processed_two = smooth_series(processed_two, window=window)
        steps.append(f"smoothed(window={window})")
        if _prompt_bool("Plot the smoothing comparison now?", default=True):
            plot_preprocessing_results(before_one, processed_one, f"Smoothing comparison: {label_one}", label_one, f"{label_one} smoothed")
            plot_preprocessing_results(before_two, processed_two, f"Smoothing comparison: {label_two}", label_two, f"{label_two} smoothed")

    if _prompt_bool("Apply downsampling before the causal analysis?", default=False):
        downsample_mode = _prompt_choice(
            "Choose a downsampling mode",
            ["every Nth observation", "calendar frequency alias (for datetime indexes)"],
            default_index=0,
        )
        before_one = processed_one
        before_two = processed_two
        if downsample_mode == 0:
            step = _prompt_int("Keep every Nth observation", 2)
            processed_one = downsample_series(processed_one, step=step)
            processed_two = downsample_series(processed_two, step=step)
            steps.append(f"downsampled(step={step})")
        else:
            freq = input("Enter pandas frequency alias (for example, W, M, or Q) [W]: ").strip() or "W"
            processed_one = downsample_series(processed_one, freq=freq)
            processed_two = downsample_series(processed_two, freq=freq)
            steps.append(f"downsampled(freq='{freq}')")
        if _prompt_bool("Plot the downsampling comparison now?", default=True):
            plot_preprocessing_results(before_one, processed_one, f"Downsampling comparison: {label_one}", label_one, f"{label_one} downsampled")
            plot_preprocessing_results(before_two, processed_two, f"Downsampling comparison: {label_two}", label_two, f"{label_two} downsampled")

    if _prompt_bool("Apply z-score standardization (subtract mean, divide by standard deviation)?", default=False):
        processed_one = standardize_series(processed_one)
        processed_two = standardize_series(processed_two)
        steps.append("z-scored")

    common_index = processed_one.index.intersection(processed_two.index)
    processed_one = processed_one.loc[common_index]
    processed_two = processed_two.loc[common_index]

    label_suffix = steps[0] if len(steps) == 1 else ", ".join(steps)
    final_label_one = f"{label_one} ({label_suffix})"
    final_label_two = f"{label_two} ({label_suffix})"
    summary_one = summarize_preprocessing(series_one, processed_one, label_one, final_label_one)
    summary_two = summarize_preprocessing(series_two, processed_two, label_two, final_label_two)
    summary_one["operations"] = steps
    summary_two["operations"] = steps
    return processed_one, processed_two, summary_one, summary_two


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
