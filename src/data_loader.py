"""Data-loading helpers for the causality toolkit.

This module centralizes the non-interactive loading primitives used by the
toolkit so workflows, notebooks, and CLI entry points can share the same data
source handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class LoadedSeriesPair:
    """Bundle two aligned 1-D series and their human-readable labels."""

    left: pd.Series
    right: pd.Series
    left_name: str
    right_name: str


def _validate_date(text: str) -> pd.Timestamp | None:
    try:
        value = pd.to_datetime(text, errors="raise")
    except Exception:
        return None
    return None if pd.isna(value) else pd.Timestamp(value)


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
) -> LoadedSeriesPair:
    """Download close-price data and return two aligned raw series."""
    tickers = [ticker_one, ticker_two]
    frame = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)
    if frame.empty:
        raise RuntimeError("Yahoo Finance returned no data for the requested tickers and date range.")

    if isinstance(frame.columns, pd.MultiIndex):
        if "Close" not in frame.columns.get_level_values(0):
            raise RuntimeError("Yahoo Finance data did not include close prices.")
        close = frame["Close"].copy()
    else:
        close = frame[["Close"]].copy() if "Close" in frame.columns else frame.copy()

    close = pd.DataFrame(close).dropna(how="any")
    if close.shape[1] != 2:
        raise RuntimeError("Failed to download two valid price series. Check the ticker symbols.")
    close.index = pd.to_datetime(close.index)

    left_name = name_one or get_ticker_name(ticker_one)
    right_name = name_two or get_ticker_name(ticker_two)

    if ticker_one in close.columns and ticker_two in close.columns:
        left_source = close[ticker_one]
        right_source = close[ticker_two]
    else:
        ordered_columns = list(close.columns)
        left_source = close[ordered_columns[0]]
        right_source = close[ordered_columns[1]]

    left = pd.Series(left_source, index=close.index, name=left_name).dropna()
    right = pd.Series(right_source, index=close.index, name=right_name).dropna()
    common_index = left.index.intersection(right.index)
    return LoadedSeriesPair(left=left.loc[common_index], right=right.loc[common_index], left_name=left_name, right_name=right_name)


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
    """Load two one-dimensional series from CSV files and pair them by index."""
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
    ).dropna()
    right_series = pd.Series(
        pd.to_numeric(right_frame[right_value_column], errors="coerce"),
        index=right_index,
        name=right_name or right_value_column,
    ).dropna()
    common_index = left_series.index.intersection(right_series.index)
    return LoadedSeriesPair(
        left=left_series.loc[common_index],
        right=right_series.loc[common_index],
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
    """Normalize two user-provided 1-D series into aligned pandas Series objects."""
    left_series = _coerce_series(left, left_name)
    right_series = _coerce_series(right, right_name)
    common_index = left_series.index.intersection(right_series.index)
    return LoadedSeriesPair(
        left=left_series.loc[common_index],
        right=right_series.loc[common_index],
        left_name=left_series.name or left_name,
        right_name=right_series.name or right_name,
    )


def download_data(ticker_one: str, ticker_two: str, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compatibility wrapper returning price data and log returns for two tickers."""
    loaded = download_yfinance_series(ticker_one, ticker_two, start, end)
    prices = pd.DataFrame({ticker_one: loaded.left, ticker_two: loaded.right})
    log_returns = pd.DataFrame({
        f"{ticker_one}_log_return": np.log(prices[ticker_one]).diff(),
        f"{ticker_two}_log_return": np.log(prices[ticker_two]).diff(),
    }).dropna()
    return prices, log_returns
