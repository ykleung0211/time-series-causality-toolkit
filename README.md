# Time-Series Causality Toolkit

This project is structured as a reusable toolkit plus case studies.

## Structure

- `finance_dtw_causality_toy/`: core reusable code
	- `data.py`: ticker lookup and data download helpers
	- `preprocessing.py`: smoothing and downsampling
	- `stationarity.py`: ADF / unit-root testing and automatic differencing
	- `metrics.py`: DTW, Granger causality, transfer entropy, CCM, and parameter sweeps
	- `surrogate.py`: shuffle and bootstrap surrogate tests with reproducible seeds
	- `plotting.py`: reusable visualization helpers
- `case_studies/`: ready-to-run examples
	- `case_studies/finance/yfinance_case.py`: interactive Yahoo Finance case study
	- `case_studies/environmental_health_case.py`: ozone concentration vs mortality example
- `case_studies/market_regime_case.py`: SPY vs VIX market-regime example
	- `case_studies/macro_regime_case.py`: inflation vs unemployment macro example
- `main.py`: thin entry point that lets you choose an example

## What the workflow does

1. Runs the finance example on two Yahoo Finance tickers, the environmental/health example on ozone versus mortality data, the market-regime example on SPY versus VIX, or the macro-regime example on inflation versus unemployment.
2. Applies optional downsampling, smoothing, and lagged cross-correlation for the non-finance examples as well.
3. Runs ADF stationarity checks and differences the series if needed.
4. Computes DTW, Granger causality, transfer entropy, and CCM.
5. Optionally searches TE/CCM parameter combinations and plots results.
6. Runs shuffle or bootstrap surrogate tests with a user-provided random seed.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python main.py
```

The finance case study is currently the main example. Additional domain-specific case studies can be added under `case_studies/` without changing the core toolkit.