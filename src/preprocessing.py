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


def preprocess_single_series(series: pd.Series, config: PreprocessingConfig) -> tuple[pd.Series, dict[str, object]]:
    """Apply preprocessing steps to a single series and return metadata about the transform."""
    processed, base_label = _apply_base_representation(series, config.base_representation)
    operations = [base_label]

    processed, smoothing_label = _apply_optional_smoothing(processed, config.smoothing_window)
    if smoothing_label is not None:
        operations.append(smoothing_label)

    processed, downsample_label = _apply_optional_downsampling(processed, config.downsample_step, config.downsample_freq)
    if downsample_label is not None:
        operations.append(downsample_label)

    if config.standardize:
        processed = standardize_series(processed)
        operations.append("z-scored")

    metadata = {
        "operations": operations,
        "base_representation": config.base_representation,
        "output_length": int(len(pd.Series(processed).dropna())),
    }
    return processed, metadata


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
    left = pd.to_numeric(pd.Series(series_one), errors="coerce").reset_index(drop=True)
    right = pd.to_numeric(pd.Series(series_two), errors="coerce").reset_index(drop=True)

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

        frame = pd.DataFrame({"left": aligned_left, "right": aligned_right}).dropna()
        if len(frame) < 2:
            correlation = np.nan
        else:
            correlation = float(frame["left"].corr(frame["right"]))
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
    config: PreprocessingConfig | None = None,
    *,
    left_config: PreprocessingConfig | None = None,
    right_config: PreprocessingConfig | None = None,
) -> PreprocessingResult:
    """Apply preprocessing independently to two series.

    The function does not prompt for input or print output. It returns the
    transformed series, final labels, and structured summaries for notebook or
    workflow consumption.
    """
    shared_config = config or PreprocessingConfig()
    effective_left_config = left_config or shared_config
    effective_right_config = right_config or shared_config

    processed_one, left_metadata = preprocess_single_series(series_one, effective_left_config)
    processed_two, right_metadata = preprocess_single_series(series_two, effective_right_config)

    left_suffix = ", ".join(left_metadata["operations"])
    right_suffix = ", ".join(right_metadata["operations"])
    final_label_one = f"{label_one} ({left_suffix})"
    final_label_two = f"{label_two} ({right_suffix})"

    summary_one = summarize_preprocessing(series_one, processed_one, label_one, final_label_one)
    summary_two = summarize_preprocessing(series_two, processed_two, label_two, final_label_two)
    summary_one["operations"] = left_metadata["operations"]
    summary_two["operations"] = right_metadata["operations"]
    return PreprocessingResult(
        left=processed_one,
        right=processed_two,
        left_label=final_label_one,
        right_label=final_label_two,
        summary_left=summary_one,
        summary_right=summary_two,
        operations=list(dict.fromkeys(left_metadata["operations"] + right_metadata["operations"])),
        config=shared_config,
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
