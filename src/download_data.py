# Data acquisition helpers for the causality toolkit.

from __future__ import annotations

import pandas as pd
import numpy as np
import yfinance as yf


def get_ticker_input() -> tuple[str, str]:
    # Prompt the user for two ticker symbols.
    print("Enter two ticker symbols to analyze. Examples: ^IXIC, ^GSPC, AAPL, MSFT.")
    t1 = input("First ticker: ").strip()
    t2 = input("Second ticker: ").strip()
    return t1, t2


def get_ticker_name(ticker: str) -> str:
    # Return a human-readable name for a ticker if available.
    try:
        info = yf.Ticker(ticker).info
        name = info.get("shortName") or info.get("longName")
        return name if name else ticker
    except Exception:
        return ticker


def get_date_range() -> tuple[str, str]:
    # Prompt for a date range and fall back to a default window.
    print("Specify analysis date range. Press Enter to use default (2015-01-01 to today).")
    start = input("Start date (YYYY-MM-DD): ").strip() or "2015-01-01"
    end = input("End date (YYYY-MM-DD): ").strip() or pd.Timestamp.today().strftime("%Y-%m-%d")

    try:
        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)
        if start_ts > end_ts:
            print("Start date is after end date - swapping.")
            start_ts, end_ts = end_ts, start_ts
        return start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d")
    except Exception:
        print("Invalid date(s). Using defaults.")
        return "2015-01-01", pd.Timestamp.today().strftime("%Y-%m-%d")


def download_data(ticker_one: str, ticker_two: str, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Download prices and return aligned log returns for two tickers.
    tickers = [ticker_one, ticker_two]
    data = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    data = pd.DataFrame(data).dropna()
    if data.shape[1] != 2:
        raise RuntimeError("Failed to download two tickers. Check symbols.")

    data.columns = [ticker_one, ticker_two]
    log_returns = np.log(data).diff().dropna()
    log_returns.columns = [f"{ticker_one}_log_return", f"{ticker_two}_log_return"]
    return data, log_returns
