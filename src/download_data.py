"""Compatibility wrapper for data-loading helpers.

The implementation now lives in :mod:`src.data_loader` so the package can keep
supporting the historical import path while the refactor lands.
"""

from __future__ import annotations

from .data_loader import (
    LoadedSeriesPair,
    coerce_two_series,
    download_data,
    download_yfinance_series,
    get_date_range,
    get_ticker_input,
    get_ticker_name,
    load_two_series_from_csv,
    prompt_optional_series_names,
    prompt_yfinance_series,
)

__all__ = [
    "LoadedSeriesPair",
    "coerce_two_series",
    "download_data",
    "download_yfinance_series",
    "get_date_range",
    "get_ticker_input",
    "get_ticker_name",
    "load_two_series_from_csv",
    "prompt_optional_series_names",
    "prompt_yfinance_series",
]
