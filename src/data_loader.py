"""Data-loading helpers for the causality toolkit.

This module centralizes the non-interactive loading primitives used by the
toolkit so workflows, notebooks, and CLI entry points can share the same data
source handling.
"""

from __future__ import annotations

from dataclasses import dataclass # define LoadedSeriesPair
from pathlib import Path
from typing import Any # Handle loosely typed series inputs for coerce_two_series

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True) # fronzen=True makes the dataclass immutable
class LoadedSeriesPair:
    """Bundle two aligned 1-D series and their human-readable labels."""

    left: pd.Series
    right: pd.Series
    left_name: str
    right_name: str


def _download_yfinance_single_series(
    ticker: str,
    start: str,
    end: str,
    *, # force keyword-only arguments, must write field=..., name=..., frequency=... when calling the function
    field: str = "Close",
    name: str | None = None,
    frequency: str | None = None,
) -> pd.Series:
    # progress=False disables the progress bar, auto_adjust=False keeps the raw data without adjusting for splits/dividends 
    frame = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if frame.empty:
        raise RuntimeError(f"Yahoo Finance returned no data for ticker '{ticker}' and the requested date range.")

    raw_field: pd.Series | pd.DataFrame | None = None
    if isinstance(frame.columns, pd.MultiIndex):
        # frame.columns.get_level_values(0) returns the first level of the MultiIndex, which is usually the field name (e.g., 'Close', 'Open', etc.)
        if field in frame.columns.get_level_values(0):
            raw_field = frame[field] # This will return a DataFrame with the field as the column

        # frame.columns.get_level_values(-1) returns the last level of the MultiIndex     
        elif field in frame.columns.get_level_values(-1):
            raw_field = frame.xs(field, axis=1, level=-1) # search for the field in the last level of the MultiIndex and return a DataFrame with the field as the column (axis=1).

    # If the columns are not a MultiIndex, we can check if the field is directly in the columns
    elif field in frame.columns:
        raw_field = frame[field]

    if raw_field is None:
        raise RuntimeError(f"Yahoo Finance data for ticker '{ticker}' did not include field '{field}'.")

    if isinstance(raw_field, pd.DataFrame):
        if raw_field.shape[1] != 1:
            raise RuntimeError(f"Yahoo Finance field '{field}' for ticker '{ticker}' resolved to multiple columns.")
        raw_field = raw_field.iloc[:, 0]

    series = pd.Series(raw_field.to_numpy(), index=pd.to_datetime(raw_field.index), name=name or get_ticker_name(ticker)).dropna()
    if frequency:
        if not isinstance(series.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            raise TypeError("Frequency-based resampling requires a DatetimeIndex or PeriodIndex.")
        series = series.resample(frequency).last().dropna() # takes the last value in each resampled period, e.g., for 'M' (monthly) it takes the last value of each month
    return series


def get_ticker_name(ticker: str) -> str:
    """Return a human-readable name for a ticker if available."""
    try:
        info = yf.Ticker(ticker).info
        name = info.get("shortName") or info.get("longName")
        return name if name else ticker
    except Exception:
        return ticker


def download_yfinance_series(
    ticker_one: str,
    ticker_two: str,
    start: str,
    end: str,
    name_one: str | None = None,
    name_two: str | None = None,
    *, # use the per-series start/end and frequency parameters to allow for non-overlapping series
    start_one: str | None = None,
    end_one: str | None = None,
    start_two: str | None = None,
    end_two: str | None = None,
    field_one: str = "Close",
    field_two: str = "Close",
    frequency_one: str | None = None,
    frequency_two: str | None = None,
) -> LoadedSeriesPair:
    """Download two Yahoo Finance series without intersecting their indices."""
    effective_start_one = start_one or start
    effective_end_one = end_one or end
    effective_start_two = start_two or start
    effective_end_two = end_two or end

    left = _download_yfinance_single_series(
        ticker_one,
        effective_start_one,
        effective_end_one,
        field=field_one,
        name=name_one or get_ticker_name(ticker_one),
        frequency=frequency_one,
    )
    right = _download_yfinance_single_series(
        ticker_two,
        effective_start_two,
        effective_end_two,
        field=field_two,
        name=name_two or get_ticker_name(ticker_two),
        frequency=frequency_two,
    )

    return LoadedSeriesPair(
        left=left,
        right=right,
        left_name=left.name or ticker_one,
        right_name=right.name or ticker_two,
    )


def load_two_series_from_csv(
    left_path: str | Path,
    right_path: str | Path,
    left_value_column: str,
    right_value_column: str,
    left_name: str | None = None,
    right_name: str | None = None,
    left_index_column: str | None = None,
    right_index_column: str | None = None,
) -> LoadedSeriesPair:
    """Load two one-dimensional series from CSV files."""
    left_frame = pd.read_csv(left_path)
    right_frame = pd.read_csv(right_path)

    if left_value_column not in left_frame.columns:
        raise ValueError(f"Column '{left_value_column}' was not found in {left_path}.")
    if right_value_column not in right_frame.columns:
        raise ValueError(f"Column '{right_value_column}' was not found in {right_path}.")

    if left_index_column is not None and left_index_column not in left_frame.columns:
        raise ValueError(f"Index column '{left_index_column}' was not found in {left_path}.")
    if right_index_column is not None and right_index_column not in right_frame.columns:
        raise ValueError(f"Index column '{right_index_column}' was not found in {right_path}.")

    left_index = pd.to_datetime(left_frame[left_index_column], errors="coerce") if left_index_column else left_frame.index
    right_index = pd.to_datetime(right_frame[right_index_column], errors="coerce") if right_index_column else right_frame.index

    left_series = pd.Series(
        pd.to_numeric(left_frame[left_value_column], errors="coerce"),
        index=left_index,
        name=left_name or left_value_column,
    )
    right_series = pd.Series(
        pd.to_numeric(right_frame[right_value_column], errors="coerce"),
        index=right_index,
        name=right_name or right_value_column,
    )

    if isinstance(left_series.index, pd.DatetimeIndex):
        left_series = left_series[left_series.index.notna()]

    if isinstance(right_series.index, pd.DatetimeIndex):
        right_series = right_series[right_series.index.notna()]

    left_series = left_series.dropna()
    right_series = right_series.dropna()
    
    return LoadedSeriesPair(
        left=left_series,
        right=right_series,
        left_name=left_series.name,
        right_name=right_series.name,
    )


def _coerce_series(value: Any, default_name: str) -> pd.Series:
    if isinstance(value, pd.Series):
        series = value.copy()
        if series.name is None:
            series.name = default_name
        return pd.to_numeric(series, errors="coerce").dropna()

    if isinstance(value, (list, tuple, np.ndarray)):
        return pd.Series(pd.to_numeric(pd.Series(value), errors="coerce"), name=default_name).dropna()

    raise TypeError("Series inputs must be pandas Series, lists, tuples, or numpy arrays.")


def coerce_two_series(
    left: Any,
    right: Any,
    left_name: str = "Series 1",
    right_name: str = "Series 2",
) -> LoadedSeriesPair:
    """Normalize two user-provided 1-D series into pandas Series objects without alignment."""
    left_series = _coerce_series(left, left_name)
    right_series = _coerce_series(right, right_name)
    
    return LoadedSeriesPair(
        left=left_series,
        right=right_series,
        left_name=left_series.name or left_name,
        right_name=right_series.name or right_name,
    )
