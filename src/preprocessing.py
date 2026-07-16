"""Preprocessing utilities for time-series analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .plotting import plot_line_trend, plot_preprocessing_results, plot_single_series


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for transforming a pair of aligned series.

    Attributes:
        base_representation: The representation to use before optional steps.
            Supported values are ``"raw"``, ``"returns"``, and ``"log_returns"``.
        smoothing_window: Optional centered rolling window size.
        downsample_step: Optional integer step for row-wise downsampling.
        downsample_freq: Optional pandas frequency alias for datetime-indexed
            series.
        standardize: Whether to z-score the final series.
    """

    base_representation: str = "log_returns"
    smoothing_window: int | None = None
    downsample_step: int | None = None
    downsample_freq: str | None = None
    standardize: bool = False


@dataclass(frozen=True)
class PreprocessingResult:
    """Structured output of a preprocessing run.

    Attributes:
        left: The processed left-hand series.
        right: The processed right-hand series.
        left_label: Final display label for the left-hand series.
        right_label: Final display label for the right-hand series.
        summary_left: Summary dictionary for the left-hand series.
        summary_right: Summary dictionary for the right-hand series.
        operations: Ordered list of transformations applied.
        config: The configuration used to produce the result.
    """

    left: pd.Series
    right: pd.Series
    left_label: str
    right_label: str
    summary_left: dict[str, object]
    summary_right: dict[str, object]
    operations: list[str]
    config: PreprocessingConfig


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


def _apply_base_representation(series: pd.Series, base_representation: str) -> tuple[pd.Series, str]:
    cleaned = pd.Series(series).dropna()
    if base_representation == "raw":
        return cleaned, "raw values"
    if base_representation == "returns":
        return compute_returns(cleaned), "percentage changes"
    if base_representation == "log_returns":
        return compute_log_returns(cleaned), "log changes"
    raise ValueError("base_representation must be one of 'raw', 'returns', or 'log_returns'.")


def _apply_optional_smoothing(series: pd.Series, window: int | None) -> tuple[pd.Series, str | None]:
    if window is None:
        return series, None
    smoothed = smooth_series(series, window=window)
    return smoothed, f"smoothed(window={window})"


def _apply_optional_downsampling(series: pd.Series, step: int | None, freq: str | None) -> tuple[pd.Series, str | None]:
    if freq:
        return downsample_series(series, freq=freq), f"downsampled(freq='{freq}')"
    if step and step > 1:
        return downsample_series(series, step=step), f"downsampled(step={step})"
    return series, None


def preprocess_series_pair(
    series_one: pd.Series,
    series_two: pd.Series,
    label_one: str,
    label_two: str,
    config: PreprocessingConfig,
) -> PreprocessingResult:
    """Apply a preprocessing configuration to two aligned series.

    The function does not prompt for input or print output. It returns the
    transformed series, final labels, and structured summaries for notebook or
    workflow consumption.
    """
    processed_one, base_label = _apply_base_representation(series_one, config.base_representation)
    processed_two, _ = _apply_base_representation(series_two, config.base_representation)
    operations = [base_label]

    processed_one, smoothing_label = _apply_optional_smoothing(processed_one, config.smoothing_window)
    processed_two, _ = _apply_optional_smoothing(processed_two, config.smoothing_window)
    if smoothing_label is not None:
        operations.append(smoothing_label)

    processed_one, downsample_label = _apply_optional_downsampling(processed_one, config.downsample_step, config.downsample_freq)
    processed_two, _ = _apply_optional_downsampling(processed_two, config.downsample_step, config.downsample_freq)
    if downsample_label is not None:
        operations.append(downsample_label)

    if config.standardize:
        processed_one = standardize_series(processed_one)
        processed_two = standardize_series(processed_two)
        operations.append("z-scored")

    common_index = processed_one.index.intersection(processed_two.index)
    processed_one = processed_one.loc[common_index]
    processed_two = processed_two.loc[common_index]

    label_suffix = operations[0] if len(operations) == 1 else ", ".join(operations)
    final_label_one = f"{label_one} ({label_suffix})"
    final_label_two = f"{label_two} ({label_suffix})"
    summary_one = summarize_preprocessing(series_one, processed_one, label_one, final_label_one)
    summary_two = summarize_preprocessing(series_two, processed_two, label_two, final_label_two)
    summary_one["operations"] = operations
    summary_two["operations"] = operations
    return PreprocessingResult(
        left=processed_one,
        right=processed_two,
        left_label=final_label_one,
        right_label=final_label_two,
        summary_left=summary_one,
        summary_right=summary_two,
        operations=operations,
        config=config,
    )


def lagged_cross_correlation_report(
    series_one: pd.Series,
    series_two: pd.Series,
    label_one: str,
    label_two: str,
    max_lag: int = 30,
) -> pd.DataFrame:
    """Compute lagged cross-correlation and produce a plot-friendly frame.

    The function does not prompt for input. Callers can inspect the returned
    frame or decide whether to plot it.
    """
    frame = lagged_cross_correlation(series_one, series_two, max_lag=max_lag)
    valid = frame["correlation"].dropna()
    if valid.empty:
        return frame

    best_index = valid.abs().idxmax()
    plot_line_trend(
        frame,
        "lag",
        ["correlation"],
        f"Lagged cross-correlation: {label_one} vs {label_two}",
        "Lag",
        "Pearson correlation",
    )
    return frame
