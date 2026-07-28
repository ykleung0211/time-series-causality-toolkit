"""Preprocessing utilities for time-series analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import Literal

import numpy as np
import pandas as pd

from .plotting import plot_line_trend


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for transforming a pair of aligned series.

    Attributes:
        base_representation: The representation to use before optional steps.
            Supported values are ``"raw"``, ``"returns"``, and ``"log_returns"``.
        smoothing_method: The method to use for smoothing.
        smoothing_window: Optional centered rolling window size.
        smoothing_sigma: Optional standard deviation for Gaussian smoothing.
        downsample_step: Optional integer step for row-wise downsampling.
        downsample_freq: Optional pandas frequency alias for datetime-indexed
            series.
        standardize: Whether to z-score the final series.
    """

    base_representation: Literal["raw", "returns", "log_returns"] = "log_returns"
    smoothing_method: str | None = None # moving average or Gaussian smoothing method
    smoothing_window: int | None = None
    smoothing_sigma: float | None = None
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

    # Order of operations: base representation -> smoothing -> downsampling -> standardization
    processed, smoothing_label = _apply_optional_smoothing(processed, config.smoothing_method, config.smoothing_window, config.smoothing_sigma)
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

    # pct_change is x_t - x_(t-1) / x_(t-1)
    return cleaned.pct_change().dropna()


def compute_log_returns(series: pd.Series) -> pd.Series:
    """Compute log returns for a strictly positive numeric series."""
    cleaned = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    positive_with_nan = cleaned.where(cleaned > 0)
    return np.log(positive_with_nan).diff().dropna()


def smooth_series(series: pd.Series, window: int = 5) -> pd.Series:
    """Apply trailing moving average smoothing to a numeric series."""
    # center=False ensures that the rolling mean is a trailing average (i.e., it only uses the current and past values, not future values).
    # when center=True, look-ahead bias would invalidate the casual interpretation of the smoothed series
    return pd.Series(series).rolling(window=window, center=False).mean().dropna()



def gaussian_smooth_series(series: pd.Series, window: int = 5, sigma: float = 1.0) -> pd.Series:
    """
    Apply trailing Gaussian smoothing to a numeric series.
    
    Unlike a standard symmetric Guassian filter, this kernel only assigns weight to the current and past observations within the window
    so it preserves the same causal (non-lookahead) property as the trailing moving average.
    Weight decays with distance into the past according to "sigma":
    a smaller sigma means more weight on the recent observations,
    while a larger sigma approaches a flatter, moving average-like weighting.
    """
    if window < 1:
        raise ValueError("Window size must be at least 1.")
    if sigma <= 0:
        raise ValueError("Sigma must be positive.")

    cleaned = pd.Series(series).dropna()
    # Create the relative positions of each point in the window, with the current point at position 0 and past points at negative positions
    offsets = np.arange(-(window - 1), 1)  # e.g., for window=5, offsets = [-4, -3, -2, -1, 0]
    # the formula for the Gaussian kernel is exp(-0.5 * (x/sigma)^2), where x is the offset from the current point
    # when offset=0, the weight is 1 (the peak of the Gaussian), and it decays for negative offsets
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()  # Normalize the kernel to sum to 1

    # np.dot is the dot product, which computes the weighted sum of the values in the window using the Gaussian kernel
    smoothed = cleaned.rolling(window=window, center=False).apply(lambda values: np.dot(values, kernel), raw=True)
    return smoothed.dropna()


def standardize_series(series: pd.Series) -> pd.Series:
    """Z-score standardize a series by subtracting the mean and dividing by the standard deviation."""
    cleaned = pd.Series(series).dropna()
    std = float(cleaned.std())
    if std == 0.0:
        # If the standard deviation is zero, all values are identical. In this case, we can return a series of zeros (or NaNs) since z-scoring would not be meaningful.
        return cleaned - cleaned.mean()
    return (cleaned - cleaned.mean()) / std


def downsample_series(series: pd.Series, step: int | None = None, freq: str | None = None) -> pd.Series:
    """Downsample by integer step or pandas frequency alias."""
    series = pd.Series(series)
    if freq:
        if not isinstance(series.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            raise TypeError("Frequency-based downsampling requires a DatetimeIndex or PeriodIndex.")

        # resample(freq) groups the data into bins of the specified frequency, and .last() takes the last value in each bin. This is a common approach for downsampling time series data.
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
            # shift the left series forward by lag, and the right series backward by lag, to compute the correlation at this lag
            aligned_left = left.iloc[:-lag].reset_index(drop=True)
            aligned_right = right.iloc[lag:].reset_index(drop=True)
        elif lag < 0:
            # shift the left series backward by lag, and the right series forward by lag, to compute the correlation at this lag
            aligned_left = left.iloc[-lag:].reset_index(drop=True)
            aligned_right = right.iloc[:lag].reset_index(drop=True)
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


def _apply_base_representation(series: pd.Series, base_representation: Literal["raw", "returns", "log_returns"]) -> tuple[pd.Series, str]:
    cleaned = pd.Series(series).dropna()
    if base_representation == "raw":
        return cleaned, "raw values"
    if base_representation == "returns":
        return compute_returns(cleaned), "percentage changes"
    if base_representation == "log_returns":
        return compute_log_returns(cleaned), "log changes"
    raise ValueError("base_representation must be one of 'raw', 'returns', or 'log_returns'.")




def _apply_optional_smoothing(series: pd.Series, method: str | None, window: int | None, sigma: float | None) -> tuple[pd.Series, str | None]:
    if window is None:
        return series, None
    if method == "gaussian":
        effective_sigma = sigma if sigma is not None else 1.0
        smoothed = gaussian_smooth_series(series, window=window, sigma=effective_sigma)
        return smoothed, f"gaussian_smoothed(window={window}, sigma={effective_sigma})"
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
        # Merge the operations from both series, ensuring uniqueness while preserving order
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

    plot_line_trend(
        frame,
        "lag",
        ["correlation"],
        f"Lagged cross-correlation: {label_one} vs {label_two}",
        "Lag",
        "Pearson correlation",
    )
    return frame
